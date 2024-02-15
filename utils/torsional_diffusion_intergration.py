import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
from rdkit import Chem
import copy
from tqdm import tqdm

from utils.z_matrix import deconstruct_z_matrix

from torsional_diffusion.utils.featurization import featurize_mol, qm9_types, dihedral_pattern 
from torsional_diffusion.utils.torsion import get_transformation_mask
from torsional_diffusion.utils.dataset import TorsionNoiseTransform
from torsional_diffusion.diffusion.sampling import InferenceDataset
from torsional_diffusion.diffusion.likelihood import divergence, inertia_tensor as compute_inertia_tensor, log_det_jac as compute_log_det_jac

"""
def generate_td_batch(mol_batch, model_name, set_name, max_conformations = None, device="cpu"):
    #NOTE: Now I am changing units here (from nm to Å)
    datapoints = []
    transform = TorsionNoiseTransform()

    for mol_dic in mol_batch:
        mol = mol_dic["rdkit_mol"]

        #create pyg data object for torsional diffusion

        data = featurize_mol(mol, qm9_types)
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            data = False

        if N < 4:
            data = False

        if data:
            data = featurize_mol(mol)

            edge_mask, mask_rotate = get_transformation_mask(data)
            if np.sum(edge_mask) < 0.5:
                data = False

            data.canonical_smi, data.mol = Chem.MolToSmiles(mol), mol
            if max_conformations:
                conformers = torch.tensor(np.load("datasets/ls_cache_data/"+model_name+"/"+set_name+"/mol_"+str(mol_dic["dataset_idx"])+".npy"))[:max_conformations]*10
                #conformers = conformers.to(device)
                data.pos = [conformer for conformer in conformers]
            else:
                conformers = torch.tensor(np.load("datasets/ls_cache_data/"+model_name+"/"+set_name+"/mol_"+str(mol_dic["dataset_idx"])+".npy"))*10
                #conformers = conformers.to(device)
                data.pos = [conformer for conformer in conformers]
            data.edge_mask = torch.tensor(edge_mask)
            data.mask_rotate = mask_rotate
            data = transform(data)
            data = data.to(device)

            td_data =  copy.deepcopy(data)

        else:
            td_data = False

        datapoints.append(td_data)
    
    return Batch(datapoints)
"""

def generate_cache_local_structures(model, dataloader, model_name, set_name):
    """_summary_

    Args:
        model_path (_type_): _description_
        dataset (_type_): _description_
        force_override (bool, optional): _description_. Defaults to False.
    """
    #call this function if not data in the correspondent folder
    #with torch.no_grad():
    for mol_batch in tqdm(dataloader):
        local_z_matrices = model.get_mode_L(mol_batch)
        for i_mol, mol_dic in enumerate(mol_batch):
            #compute indexes of rotable bonds in the z-matrix.
            mol_torsion_1_indexes = []
            mol_torsion_2_indexes = []
            for group in mol_dic["groups"]:
                if len(group) == 4:
                    mol_torsion_1_indexes.append(group[3])

                if len(group) == 5:
                    mol_torsion_2_indexes.append([ group[3], group[4] ])
            list_torsion_2_indexes = []
            for pair in mol_torsion_2_indexes:
                list_torsion_2_indexes.append(pair[0])
                list_torsion_2_indexes.append(pair[1])
            rb_atoms = [int(i) for i in range(3, len(mol_dic["atoms"])) if i not in mol_torsion_1_indexes and i not in list_torsion_2_indexes]
            rb_torsion_indexes = [i-1 for i in rb_atoms]
            mol_z_matrices = torch.cat([local_z_matrices[i_mol].unsqueeze(dim=0)]*mol_dic["int_coord"].size()[0], dim = 0).detach().to("cpu")
            if rb_torsion_indexes:
                mol_z_matrices[:, rb_torsion_indexes,2] = mol_dic["int_coord"][:, rb_torsion_indexes,2]
            mol_confs = torch.zeros((mol_z_matrices.size()[0], mol_z_matrices.size()[1]+1, 3))
            for i, z_matrix in enumerate(mol_z_matrices):
                mol_confs[i] = deconstruct_z_matrix(z_matrix, mol_dic["ref_atoms"], loss=False)
            mol_confs = mol_confs.detach().numpy()
            np.save("datasets/ls_cache_data/"+model_name+"/"+set_name+"/cc_"+str(mol_dic["dataset_idx"])+".npy", mol_confs)
    return

class TDDataset(Dataset):
    def __init__(self, base_dataset, transform=None, num_workers=1, limit_molecules=None,
                 cache=None, boltzmann_resampler=None, max_conformations = None):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol

        super(TDDataset, self).__init__(transform=transform)

        self.base_dataset = base_dataset
        self.boltzmann_resampler = boltzmann_resampler
        self.max_conformations = max_conformations

        if limit_molecules:
            self.datapoints = self.datapoints[:limit_molecules]

    def len(self):
        return len(self.base_dataset)

    def get(self, idx):
        
        mol_dic = self.base_dataset[idx]
        mol = mol_dic["rdkit_mol"]

        #create pyg data object for torsional diffusion

        data = featurize_mol(mol, qm9_types)
        N = mol.GetNumAtoms()
        #if not mol.HasSubstructMatch(dihedral_pattern):
        #    return None

        #if N < 4:
        #    return None

        if data:

            edge_mask, mask_rotate = get_transformation_mask(data)
            #if np.sum(edge_mask) < 0.5:
                #return None

            data.canonical_smi, data.mol = Chem.MolToSmiles(mol), mol
            if self.max_conformations:
                conformers = torch.tensor(mol_dic["conformations"])[:self.max_conformations]*10 
                data.pos = [conformer for conformer in conformers]
            else:
                conformers = torch.tensor(mol_dic["conformations"])*10
                data.pos = [conformer for conformer in conformers] #make this a random assignment to be faster?
            data.edge_mask = torch.tensor(edge_mask)
            data.mask_rotate = mask_rotate
            data.dataset_index = [idx]

        if self.boltzmann_resampler:
            self.boltzmann_resampler.try_resample(data)
            
        return data

def generate_pyg_data_from_rdkit_mol(rd_mol):
    """Creates a pyg data objectfrom an rdkit mol

    Args:
        rd_mol (_type_): _description_
    """
    data = featurize_mol(rd_mol, qm9_types)
    edge_mask, mask_rotate = get_transformation_mask(data)
    data.canonical_smi, data.mol = Chem.MolToSmiles(rd_mol), rd_mol

    data.edge_mask = torch.tensor(edge_mask)
    data.mask_rotate = mask_rotate

    return data

def sample_rotable_bonds(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None, device = "cpu", still_frames = 10):
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    for batch_idx, data in enumerate(loader):

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):

            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()

            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood)
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)

            if pdb:
                for conf_idx in range(data.num_graphs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()

    return conformers


def populate_cartesian_likelihood(data):
    try:
        dlogp = data.dlogp
    except:
        dlogp = 0
    inertia_tensor = compute_inertia_tensor(data.pos)
    log_det_jac = compute_log_det_jac(data)
    data.euclidean_dlogp = dlogp - 0.5 * np.log(np.abs(np.linalg.det(inertia_tensor))) - log_det_jac

    return

import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
import numpy as np
from e3nn.nn import BatchNorm
import torsional_diffusion.diffusion.torus as torus

def get_td_model(args):
    return TensorProductScoreModel(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                                   ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                                   sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                   num_conv_layers=args.num_conv_layers,
                                   max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                                   scale_by_sigma=args.scale_by_sigma,
                                   use_second_order_repr=args.use_second_order_repr,
                                   residual=not args.no_residual, batch_norm=not args.no_batch_norm)


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, in_node_features=74, in_edge_features=4, sigma_embed_dim=32, sigma_min=0.01 * np.pi,
                 sigma_max=np.pi, sh_lmax=2, ns=32, nv=8, num_conv_layers=4, max_radius=5, radius_embed_dim=50,
                 scale_by_sigma=True, use_second_order_repr=True, batch_norm=True, residual=True
                 ):
        super(TensorProductScoreModel, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma

        self.node_embedding = nn.Sequential(
            nn.Linear(in_node_features + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_features + sigma_embed_dim + radius_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )
        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)
        conv_layers = []

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.final_edge_embedding = nn.Sequential(
            nn.Linear(radius_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )

        self.final_tp = o3.FullTensorProduct(self.sh_irreps, "2e")

        self.bond_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps,
            sh_irreps=self.final_tp.irreps_out,
            out_irreps=f'{ns}x0o',
            n_edge_features=3 * ns,
            residual=False,
            batch_norm=batch_norm
        )

        self.final_linear = nn.Sequential(
            nn.Linear(ns, ns, bias=False),
            nn.Tanh(),
            nn.Linear(ns, 1, bias=False)
        )

    def forward(self, data):
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
        src, dst = edge_index

        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.conv_layers:
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')

        bonds, edge_index, edge_attr, edge_sh = self.build_bond_conv_graph(data, node_attr)

        bond_vec = data.pos[bonds[1]] - data.pos[bonds[0]]
        bond_attr = node_attr[bonds[0]] + node_attr[bonds[1]]

        bonds_sh = o3.spherical_harmonics("2e", bond_vec, normalize=True, normalization='component')
        edge_sh = self.final_tp(edge_sh, bonds_sh[edge_index[0]])

        edge_attr = torch.cat([edge_attr, node_attr[edge_index[1], :self.ns], bond_attr[edge_index[0], :self.ns]], -1)
        out = self.bond_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=data.edge_mask.sum(), reduce='mean')

        out = self.final_linear(out)

        data.edge_pred = out.squeeze()
        data.edge_sigma = data.node_sigma[data.edge_index[0]][data.edge_mask]

        if self.scale_by_sigma:
            score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
            score_norm = torch.tensor(score_norm, device=data.x.device)
            data.edge_pred = data.edge_pred * torch.sqrt(score_norm)
        return data
    
    def generate_embeddings(self, data):
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
        src, dst = edge_index

        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.conv_layers:
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')
        edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
        #potentially add extra layer: node_attr -> n_emb scalars.
            
        return node_attr[:, :self.ns]

    def build_bond_conv_graph(self, data, node_attr):

        bonds = data.edge_index[:, data.edge_mask].long()
        bond_pos = (data.pos[bonds[0]] + data.pos[bonds[1]]) / 2
        bond_batch = data.batch[bonds[0]]
        edge_index = radius(data.pos, bond_pos, self.max_radius, batch_x=data.batch, batch_y=bond_batch)

        edge_vec = data.pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

    def build_conv_graph(self, data):

        radius_edges = radius_graph(data.pos, self.max_radius, data.batch)
        edge_index = torch.cat([data.edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data.edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_edge_features, device=data.x.device)
        ], 0)

        node_sigma = torch.log(data.node_sigma / self.sigma_min) / np.log(self.sigma_max / self.sigma_min) * 10000
        node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim)

        edge_sigma_emb = node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data.x, node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
