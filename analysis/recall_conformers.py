import os
import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdMolAlign import  GetBestRMS

from parameters import params
from utils.mdqm9_loader import MDQM9EvalDataset
from utils.torsional_clustering import get_conformers
from utils.z_matrix import construct_z_matrix_batch

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

p = Struct(**params)

dataset = MDQM9EvalDataset(p.sdf_path, p.hdf5_path)

source = "md_tuned"
model_name = "dls_r_3"
reference = "re"
indices = np.load("datasets/mdqm9-nc/splits/test_indices.npy")[:100]

directories = ["metrics", "metrics/rmsd/", "metrics/rmsd/recall/", f"metrics/rmsd/recall/{source}_{reference}/"]
if source == "md_tuned":
    directories.append(f"metrics/rmsd/recall/{source}_{reference}/{model_name}/")
for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)

for dataset_idx in tqdm(indices):
    dataset_idx = int(dataset_idx)
    mol_dic = dataset[dataset_idx]
    #print(f"Evaluating molecule {test_idx} of the test set with dataset index {dataset_idx}...")
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
    rb_atoms = [group[2] for group in mol_dic["groups"][1:]]
    
    if source == "md_tuned":
        source_path = "sampled_data/conformations/md_tuned/"+model_name+"/{:0=5}_1_ns_rw.npy".format(dataset_idx)
        source_positions = np.load(source_path)
    elif source == "md":
        source_positions = mol_dic["conformations"]
    elif source == "md_rt":
        source_positions = mol_dic["mdrt_conformations"]   
    else:
        print("Source not recognized.", flush=True)
        break 
        
    if reference == "re":
        reference_positions = mol_dic["re_conformations"]
    else:
        print("Reference not recognized.", flush=True)
        break
        
    source_z_matrices = construct_z_matrix_batch(torch.tensor(source_positions), mol_dic["ref_atoms"], list(range(source_positions.shape[1]))).detach().numpy()
    source_z_matrices = source_z_matrices[np.where([ not np.any(np.isnan(conf)) for conf in source_z_matrices])[0],:,:]
    
    referece_z_matrices = construct_z_matrix_batch(torch.tensor(reference_positions), mol_dic["ref_atoms"], list(range(reference_positions.shape[1]))).detach().numpy()
    referece_z_matrices = referece_z_matrices[np.where([ not np.any(np.isnan(conf)) for conf in referece_z_matrices])[0],:,:]

    source_z_matrices[:,:,0] = source_z_matrices[:,:,0]*10
    referece_z_matrices[:,:,0] = referece_z_matrices[:,:,0]*10

    source_conformers, md_atoms = get_conformers(source_z_matrices, rb_atoms, mol_dic, threshold=0.05)
    reference_conformers, gen_atoms = get_conformers(referece_z_matrices, rb_atoms, mol_dic, threshold=0.05)

    reference_conformers = reference_conformers.astype("double")
    source_conformers = source_conformers.astype("double")

    mol_1 = mol_dic["rdkit_mol"]
    mol_2 = Chem.Mol(mol_1)

    min_rmsd = []
    min_indices = []

    for i_conf, reference_conformer in enumerate(reference_conformers):

        conf_2 = mol_2.GetConformer()
        for i in range(mol_2.GetNumAtoms()):
            x,y,z = reference_conformer[i]
            conf_2.SetAtomPosition(i,Point3D(x,y,z))

        rmsd = []
        for source_conformer in source_conformers:
            conf_1 = mol_1.GetConformer()
            for i in range(mol_1.GetNumAtoms()):
                x,y,z = source_conformer[i]
                conf_1.SetAtomPosition(i,Point3D(x,y,z))
            rmsd.append(GetBestRMS(mol_1, mol_2))
        min_index = np.argmin(rmsd.copy())
        min_indices.append(min_index)
        min_rmsd.append(rmsd[min_index])
    
    min_rmsd = np.array(min_rmsd)
    min_indices = np.array(min_indices)
    #trying inverted structure as well
    min_rmsd_inverse = []
    min_indices_inverse = []
    for reference_conformer in reference_conformers:
        conf_2 = mol_2.GetConformer()
        for i in range(mol_2.GetNumAtoms()):
            x,y,z = reference_conformer[i]
            conf_2.SetAtomPosition(i,Point3D(x,y,z))
        rmsd = []
        for source_conformer in source_conformers:
            conf_1 = mol_1.GetConformer()
            for i in range(mol_1.GetNumAtoms()):
                x,y,z = -source_conformer[i]
                conf_1.SetAtomPosition(i,Point3D(x,y,z))
            rmsd.append(GetBestRMS(mol_1, mol_2))
        min_index = np.argmin(rmsd.copy())
        min_indices_inverse.append(min_index)
        min_rmsd_inverse.append(rmsd[min_index])
    
    min_rmsd_inverse = np.array(min_rmsd_inverse)
    min_indices_inverse = np.array(min_indices_inverse)
    inverse_indexes = np.where(min_rmsd_inverse<min_rmsd)
    min_rmsd[inverse_indexes] = min_rmsd_inverse[inverse_indexes]
    min_indices[inverse_indexes] = min_indices_inverse[inverse_indexes]
    if source == "md_tuned":
        np.save("metrics/rmsd/recall/{:s}_{:s}/{:s}/min_rmsd_{:0>5d}.npy".format(source, reference, model_name, dataset_idx), min_rmsd)
        np.save("metrics/rmsd/recall/{:s}_{:s}/{:s}/min_indices_{:0>5d}.npy".format(source, reference, model_name, dataset_idx), min_indices)
    else:
        np.save("metrics/rmsd/recall/{:s}_{:s}/min_rmsd_{:0>5d}.npy".format(source, reference, dataset_idx), min_rmsd)
        np.save("metrics/rmsd/recall/{:s}_{:s}/min_indices_{:0>5d}.npy".format(source, reference, dataset_idx), min_indices)