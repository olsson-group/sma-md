import os
import random
import time
import copy
import sys

import numpy as np
from tqdm import tqdm
import torch


from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as TGDataLoader


from utils.mdqm9_loader import MDQM9Dataset
from utils.mol_geometry import compute_torsion
from utils.z_matrix import construct_z_matrix_batch, deconstruct_z_matrix_batch, compute_jacobian_batch
from utils.torsional_diffusion_intergration import generate_pyg_data_from_rdkit_mol, TDDataset
import torsional_diffusion.diffusion.torus as torus
from utils.torsional_diffusion_intergration import get_td_model
from torsional_diffusion.utils.dataset import TorsionNoiseTransform
from torsional_diffusion.diffusion.sampling import sample

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
class SMA_MD:
    def __init__(self, params):
        self.params = params

    def build_model(self):
            """
            Builds a SMA-MD model.

            This method prepares the data, creates the dataset, splits the data into train and validation sets,
            creates data samplers and loaders, and builds the model for the SMA-MD algorithm.

            Returns:
                None
            """
            
            p = self.params

            #Setting a fixed seed for reproducibility
            torch.manual_seed(p.random_seed)
            random.seed(p.random_seed)

            print('Preparing data...', end = ' ', flush = True)
            sys.stdout.flush()

            #Create dataset
            self.dataset = MDQM9Dataset(p.sdf_path, p.aux_hdf5_path)
            #Split data into train validation and test set
            splits_path = p.splits_path
            if os.path.isfile(splits_path+"train_indices.npy") and os.path.isfile(splits_path+"val_indices.npy") and os.path.isfile(splits_path+"test_indices.npy"):
                print("Reading splits from "+splits_path, flush = True, end = ' ')
                self.train_indices = np.load(splits_path+"train_indices.npy")
                self.val_indices = np.load(splits_path+"val_indices.npy")
                self.test_indices = np.load(splits_path+"test_indices.npy")
            else:
                print("Unable to detect splits. Generating random splits...", end = '', flush=True)
                indices = [i for i in list(range(len(self.dataset))) if i not in p.keep_for_test]
                split_1 = int(np.floor(p.val_set_size * len(indices)))
                split_2 =  int(np.floor( (p.val_set_size + p.test_set_size) * len(indices)))
                np.random.seed(p.random_seed)
                np.random.shuffle(indices)
                self.train_indices, self.val_indices, self.test_indices = indices[split_2:], indices[:split_1], indices[split_1:split_2]

            print('Done!', flush = True)

            print('Building the model...', end = '', flush = True)
            sys.stdout.flush()
            self.rb_model = get_td_model(p).to(p.device)

            self.ls_min_val_loss = float('inf')
            self.rb_min_val_loss = float('inf')
            self.ls_training_metrics = {'train_loss' : [], 'val_loss' : [] }
            self.rb_training_metrics = {'train_loss' : [], 'val_loss' : [] }

            print("Done!", flush = True)
    
    def train(self, save_every_epoch = False):
        """
        Trains the model using the specified parameters.

        Args:
            save_every_epoch (bool, optional): Whether to save the model and optimizer state after each epoch. 
                                               Defaults to False.

        Returns:
            None
        """
        
        p = self.params

        #Setting a fixed seed for reproducibility
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        directories = ["output_files/", "output_files/trainings/", "output_files/trainings/"+p.model_name]
        for dir in directories: 
            if not os.path.isdir(dir):
                os.mkdir(dir)
        np.save("output_files/trainings/"+p.model_name+"/train_indices.npy", self.train_indices)
        np.save("output_files/trainings/"+p.model_name+"/val_indices.npy", self.val_indices)
        np.save("output_files/trainings/"+p.model_name+"/test_indices.npy", self.test_indices)

        #Create datasets and loaders
        transform = TorsionNoiseTransform(sigma_min=p.sigma_min, sigma_max=p.sigma_max, boltzmann_weight=p.boltzmann_weight, n_conformations=p.conf_batch_size)
        self.td_dataset =  TDDataset(self.dataset, transform=transform, num_workers=p.rb_num_workers, limit_molecules=None,
                                     cache=None, boltzmann_resampler=None, max_conformations = p.rb_max_conformations)
        
        print("Filtering molecules...", end = '', flush=True)
        train_indices = [i for i in self.train_indices if len(self.dataset[int(i)]["groups"]) > 1]
        td_train_sampler = SubsetRandomSampler(train_indices)
        val_indices = [i for i in self.val_indices if len(self.dataset[int(i)]["groups"]) > 1]
        td_val_sampler = SubsetRandomSampler(val_indices)
        self.td_train_loader = TGDataLoader(dataset=self.td_dataset, batch_size=p.mol_batch_size, shuffle = False, sampler = td_train_sampler)
        self.td_val_loader = TGDataLoader(dataset=self.td_dataset, batch_size=p.mol_batch_size, shuffle = False, sampler = td_val_sampler)
        print("Done!", flush = True)
        
        #Define optimizer 
        try: self.rb_optimizer
        except:
            print("Initializing optimizer...", flush=True)
            self.rb_optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, self.rb_model.parameters()), lr = p.rb_learning_rate, weight_decay = p.rb_weight_decay)

        #Create learning rate scheduler
        try: self.rb_scheduler
        except:
            print("Initializing scheduler...", flush=True)
            self.rb_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.rb_optimizer, mode='min', factor=p.rb_lr_factor,
                                                               patience=p.rb_scheduler_patience, min_lr=p.rb_min_rel_lr)
 
        #Create a folder for the best model and training metrics
        directories = ["models/", "metrics/", "metrics/models/"]
        for dir in directories:
            if not os.path.isdir(dir):
                os.mkdir(dir)

        if save_every_epoch:
            if not os.path.isdir("models/"+p.model_name+"_per_epoch"):
                os.mkdir("models/"+p.model_name+"_per_epoch/")
        
        self.rb_model.train()
        for epoch in range(1, p.rb_epochs + 1):
            loss_tot = 0
            val_loss_tot = 0
            base_tot = 0
            val_base_tot = 0
            
            #Training
            for data in tqdm(self.td_train_loader):
                self.rb_optimizer.zero_grad()
                data = data.to(p.device)
                data = self.rb_model(data)
                pred = data.edge_pred

                score = torus.score(
                    data.edge_rotate.cpu().numpy(),
                    data.edge_sigma.cpu().numpy())
                score = torch.tensor(score, device=pred.device)
                score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
                score_norm = torch.tensor(score_norm, device=pred.device)
                loss = ((score - pred) ** 2 / score_norm).mean()

                loss.backward()
                self.rb_optimizer.step()
                loss_tot += loss.item()
                base_tot += (score ** 2 / score_norm).mean().item()

            train_loss_avg = loss_tot / (len(self.td_train_loader))
            train_base_avg = base_tot / (len(self.td_train_loader))

            #Evaluation
            self.rb_model.eval()
            with torch.no_grad():
                for data in tqdm(self.td_val_loader): 
                    data = data.to(p.device)
                    data = self.rb_model(data)
                    pred = data.edge_pred

                    score = torus.score(
                        data.edge_rotate.cpu().numpy(),
                        data.edge_sigma.cpu().numpy())
                    score = torch.tensor(score, device=pred.device)
                    score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
                    score_norm = torch.tensor(score_norm, device=pred.device)
                    val_loss = ((score - pred) ** 2 / score_norm).mean()

                    val_loss_tot += val_loss.item()
                    val_base_tot += (score ** 2 / score_norm).mean().item()

            val_loss_avg = val_loss_tot / (len(self.td_val_loader))
            val_base_avg = val_base_tot / (len(self.td_val_loader))

            self.rb_scheduler.step(val_loss_avg-val_base_avg) 


            print("Epoch {}: Training Loss {}  training base loss {}  valition loss {}, validation base loss {}".format(epoch, train_loss_avg, train_base_avg, val_loss_avg, val_base_avg), flush = True)
            self.rb_training_metrics['train_loss'].append(train_loss_avg)
            self.rb_training_metrics['val_loss'].append(val_loss_avg)
            np.save('metrics/models/train_loss_' + p.model_name + '.npy', self.rb_training_metrics['train_loss'])
            np.save('metrics/models/val_loss_' + p.model_name + '.npy', self.rb_training_metrics['val_loss'])
            if val_loss_avg < self.rb_min_val_loss:
                self.rb_min_val_loss = val_loss_avg
                torch.save( self.rb_model.state_dict(), 'models/' + p.model_name )
            if save_every_epoch:
                torch.save( self.rb_model.state_dict(), 'models/' + p.model_name+"_per_epoch/"+"/"+p.model_name+"_epoch_"+str(epoch) )
                torch.save( self.rb_optimizer.state_dict(), 'models/' + p.model_name+"_per_epoch/"+"/"+p.model_name+"_epoch_"+str(epoch)+"_optimizer")
                torch.save( self.rb_scheduler.state_dict(), 'models/' + p.model_name+"_per_epoch/"+"/"+p.model_name+"_epoch_"+str(epoch)+"_scheduler")
              
    def generate_conformations(self, mol_index, n_confs, batch_size, device, ode = True, compute_log_prob = False, time_profile = False, diffusion_steps = 20):
        '''
        Generate conformations for a given molecule.

        Parameters:
            mol_index (int): The index of the molecule in the dataset.
            n_confs (int): The number of conformations to generate.
            batch_size (int): The batch size for sampling.
            device (str): The device to run the computations on.
            ode (bool, optional): Whether to use ODE integration for sampling. Defaults to True.
            compute_log_prob (bool, optional): Whether to compute the log probability of the conformations. Defaults to False.
            time_profile (bool, optional): Whether to print the time taken for each step. Defaults to False.
            diffusion_steps (int, optional): The number of diffusion steps for sampling rotatable bonds. Defaults to 20.

        Returns:
            numpy.ndarray: The generated conformations.
            numpy.ndarray (optional): The log probability of the conformations if compute_log_prob is True.
        '''
        if time_profile:
            t0 = time.time()
        p = self.params

        if compute_log_prob:
            ode = True

        #Setting a fixed seed for reproducibility
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        #Create datasets and loaders
        mol_dic = self.dataset[mol_index]
        mol = mol_dic["rdkit_mol"] 
        data = generate_pyg_data_from_rdkit_mol(mol)

        rotatable_bonds_pos_entries = [group[2] for group in mol_dic["groups"][1:]]
        rotatable_bonds_z_entries = [group[2]-1 for group in mol_dic["groups"][1:]]
        ref_atoms = mol_dic["ref_atoms"]
        
        dls = torch.tensor(mol_dic["conformations"])
        indexes = torch.randint(len(dls), (n_confs,))
        dls = dls[indexes]
        z_matrices = construct_z_matrix_batch(dls, ref_atoms)
        z_matrices[:,:,0] *= 10
        chiral_inv_groups = []
        chiral_inv_torsions = []
        for i_group, group in enumerate(mol_dic["groups"]):
            if len(group)==4:
                central_atom = mol.GetAtomWithIdx(int(group[1]))
                if str(central_atom.GetHybridization()) == "SP3":
                    chiral_inv_groups.append(i_group)
                    chiral_inv_torsions.append(group[3]-1)
        if len(chiral_inv_torsions):                
            z_matrices[:,torch.tensor(chiral_inv_torsions), 2] = 2.*torch.pi/3 * torch.tensor(np.random.choice([-1,1], size=(n_confs,len(chiral_inv_torsions)), p = [0.5,0.5]))  #chiral inv
        z_matrices[:, rotatable_bonds_z_entries, 2] = -torch.pi + 2*torch.pi*torch.rand((n_confs, len(rotatable_bonds_z_entries)))
        positions = deconstruct_z_matrix_batch(z_matrices, ref_atoms)

        if time_profile:
            t12 = time.time()
            print(f"Generate dls: {t12-t0}.", flush=True)

        conformers = []
        for pos in positions:
            pos = pos.detach().cpu().numpy()

            data_copy = copy.deepcopy(data)
            data_copy.pos = torch.from_numpy(pos.astype(np.float32))
            data_copy.seed_mol = copy.deepcopy(mol)
            conformers.append(data_copy)        

        if time_profile:
            t2 = time.time()
            print(f"Generate TD seeds: {t2-t12}.", flush=True)
        n_rotatable_bonds = int(data.edge_mask.sum())

        if n_rotatable_bonds > 0.5:
            if not diffusion_steps:
                diffusion_steps = p.inference_steps
            if compute_log_prob:
                conformers = sample(conformers, self.rb_model, p.sigma_max, p.sigma_min, diffusion_steps, batch_size, likelihood="full", ode = True)
                if time_profile:
                    t3 = time.time()
                    print(f"Time for sampling rotatable bonds: {t3-t2} s.", flush=True)
            else:
                conformers = sample(conformers, self.rb_model, p.sigma_max, p.sigma_min, diffusion_steps, batch_size, ode = ode)
                if time_profile:
                    t3 = time.time()
                    print(f"Time for sampling rotatable: {t3-t2} s.", flush=True)

        positions = torch.zeros((z_matrices.size()[0], z_matrices.size()[1]+1, 3))
        if compute_log_prob and n_rotatable_bonds > 0.5:
            rb_logps = []
        for i_conf, conformer in enumerate(conformers):
            positions[i_conf] = conformer.pos
            if compute_log_prob and n_rotatable_bonds > 0.5:
                rb_logps.append(conformer.dlogp)
                
        ref_indexes = ref_atoms[rotatable_bonds_pos_entries]
        i1, i2, i3 = [triplet[2] for triplet in ref_indexes], [triplet[1] for triplet in ref_indexes], [triplet[0] for triplet in ref_indexes]
        x1, x2, x3, x4 = positions[:, i1, :], positions[:, i2, :], positions[:, i3, :], positions[:, rotatable_bonds_pos_entries, :]
        x1, x2, x3, x4 = torch.reshape(x1, (n_confs*len(rotatable_bonds_pos_entries),3)), torch.reshape(x2, (n_confs*len(rotatable_bonds_pos_entries),3)), torch.reshape(x3, (n_confs*len(rotatable_bonds_pos_entries),3)), torch.reshape(x4, (n_confs*len(rotatable_bonds_pos_entries),3))
        rb_torsions = torch.reshape(compute_torsion(x1, x2, x3, x4), (n_confs, len(rotatable_bonds_z_entries)))
        z_matrices[:,:,0] /= 10.
        z_matrices = z_matrices.to(device)
        z_matrices[:,rotatable_bonds_z_entries,2] = rb_torsions.to(device)
        if compute_log_prob:
            log_prob = np.copy(rb_logps)
            detj = compute_jacobian_batch(z_matrices, ref_atoms)
            log_prob -= detj.detach().cpu().numpy()
            if time_profile:
                    t4 = time.time()
                    print(f"Time for computing cartesian log probs: {t4-t3} s.", flush=True)
            return positions.detach().cpu().numpy()/10., log_prob
        else:
            return positions.detach().cpu().numpy()/10.