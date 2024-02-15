
import os
import torch
import numpy as np
from tqdm import tqdm
import sys

from parameters import params

from parameters import params
from utils.mdqm9_loader import MDQM9EvalDataset
from utils.z_matrix import construct_z_matrix_batch

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

p = Struct(**params)

mol_indices = np.load(sys.argv[1])
mode = sys.argv[2]
model_name = sys.argv[3]

dataset = MDQM9EvalDataset(p.sdf_path, p.hdf5_path)

delta_mean_d = []
delta_mean_a = []
delta_std_d = []
delta_std_a = []
delta_mean_t = []
delta_std_t = []

# Calculate the mean and standard deviation differences between reference and generated z-matrices
for dataset_idx in tqdm(mol_indices):
    mol_dic = dataset[int(dataset_idx)]

    rd_mol = mol_dic["rdkit_mol"]
    partial_charges = mol_dic["partial_charges"]
    md_z_matrices = construct_z_matrix_batch(torch.tensor(mol_dic["conformations"]), mol_dic["ref_atoms"], list(range(mol_dic["conformations"].shape[1]))).detach().numpy()
    md_z_matrices[:,:,0] = md_z_matrices[:,:,0]*10
    filename = "{:0=5}".format(dataset_idx)+"_1_ns_rw"
    sampled_positions_filename = "sampled_data/conformations/md_tuned/"+model_name+"/"+filename+".npy"
    if os.path.isfile(sampled_positions_filename):
        sampled_positions = np.load(sampled_positions_filename)*10
    else:
        print("File not found: ", sampled_positions_filename, flush=True)
        continue
    gen_z_matrices = construct_z_matrix_batch(torch.tensor(sampled_positions), mol_dic["ref_atoms"], list(range(sampled_positions.shape[1]))).detach().numpy()

    mol_torsion_1_sp2_indexes = []
    mol_torsion_1_sp3_indexes = []
    mol_torsion_2_indexes = []
    for group in mol_dic["groups"]:
        if len(group) == 4:
            central_atom = rd_mol.GetAtomWithIdx(int(group[1]))
            if str(central_atom.GetHybridization()) == "SP2":
                mol_torsion_1_sp2_indexes.append(group[3]-1)
            if str(central_atom.GetHybridization()) == "SP3":
                mol_torsion_1_sp3_indexes.append(group[3]-1)

        if len(group) == 5:
            mol_torsion_2_indexes.append([ group[3]-1, group[4]-1 ])
    list_torsion_2_indexes = []
    for pair in mol_torsion_2_indexes:
        list_torsion_2_indexes.append(pair[0])
        list_torsion_2_indexes.append(pair[1])
    
    mol_torsion_1_sp2_indexes = np.array(mol_torsion_1_sp2_indexes)
    mol_torsion_1_sp3_indexes = np.array(mol_torsion_1_sp3_indexes)
    list_torsion_2_indexes = np.array(list_torsion_2_indexes)
     
    mean_ds_md = np.mean(md_z_matrices[:,:,0], axis =0)
    std_ds_md = np.std(md_z_matrices[:,:,0], axis =0, ddof=1)
    mean_as_md =  np.mean(md_z_matrices[:,1:,1], axis =0)
    std_as_md = np.std(md_z_matrices[:,1:,1], axis =0, ddof=1)
    
    mean_ts_md = np.array([])
    std_ts_md = np.array([])
    if len(mol_torsion_1_sp2_indexes):
        t1_sp2_ts = md_z_matrices[:,mol_torsion_1_sp2_indexes,2]
        t1_sp2_ts[t1_sp2_ts<0] = t1_sp2_ts[t1_sp2_ts<0]+2*np.pi
        mean_ts_md = np.concatenate( (mean_ts_md, np.mean(t1_sp2_ts, axis=0)) )
        std_ts_md = np.concatenate( (std_ts_md, np.std(t1_sp2_ts, axis=0, ddof=1)) )
        
    if len(mol_torsion_1_sp3_indexes):
        t1_sp3_ts = np.abs(md_z_matrices[:,mol_torsion_1_sp3_indexes,2])
        mean_ts_md = np.concatenate( (mean_ts_md, np.mean(t1_sp3_ts, axis=0)) )
        std_ts_md = np.concatenate( (std_ts_md, np.std(t1_sp3_ts, axis=0, ddof=1)) )
        
    if len(list_torsion_2_indexes):
        t2_ts = md_z_matrices[:,list_torsion_2_indexes,2]
        mean_ts_md = np.concatenate( (mean_ts_md, np.mean(t2_ts, axis=0)) )
        std_ts_md = np.concatenate( (std_ts_md, np.std(t2_ts, axis=0, ddof=1)) )
    
    mean_ds_gen = np.mean(gen_z_matrices[:,:,0], axis =0)
    std_ds_gen = np.std(gen_z_matrices[:,:,0], axis =0, ddof=1)
    mean_as_gen = np.mean(gen_z_matrices[:,1:,1], axis =0)
    std_as_gen = np.std(gen_z_matrices[:,1:,1], axis =0, ddof=1)
    
    mean_ts_gen = np.array([])
    std_ts_gen = np.array([])
    if len(mol_torsion_1_sp2_indexes):
        t1_sp2_ts = gen_z_matrices[:,mol_torsion_1_sp2_indexes,2]
        t1_sp2_ts[t1_sp2_ts<0] = t1_sp2_ts[t1_sp2_ts<0]+2*np.pi
        mean_ts_gen = np.concatenate( (mean_ts_gen, np.mean(t1_sp2_ts, axis=0)) )
        std_ts_gen = np.concatenate( (std_ts_gen, np.std(t1_sp2_ts, axis=0, ddof=1)) )
        
    if len(mol_torsion_1_sp3_indexes):
        t1_sp3_ts = np.abs(gen_z_matrices[:,mol_torsion_1_sp3_indexes,2])
        mean_ts_gen = np.concatenate( (mean_ts_gen, np.mean(t1_sp3_ts, axis=0)) )
        std_ts_gen = np.concatenate( (std_ts_gen, np.std(t1_sp3_ts, axis=0, ddof=1)) )
        
    if len(list_torsion_2_indexes):
        t2_ts = gen_z_matrices[:,list_torsion_2_indexes,2]
        mean_ts_gen = np.concatenate( (mean_ts_gen, np.mean(t2_ts, axis=0)) )
        std_ts_gen = np.concatenate( (std_ts_gen, np.std(t2_ts, axis=0, ddof=1)) )
    
    delta_mean_d.append(np.abs((mean_ds_md-mean_ds_gen)/mean_ds_md))
    delta_mean_a.append(np.abs((mean_as_md-mean_as_gen)/mean_as_md))
    delta_mean_t.append(np.abs((mean_ts_md-mean_ts_gen)/mean_ts_md))
    
    delta_std_d.append(np.abs((std_ds_md-std_ds_gen)/std_ds_md))
    delta_std_a.append(np.abs((std_as_md-std_as_gen)/std_as_md))
    delta_std_t.append(np.abs((std_ts_md-std_ts_gen)/std_ts_md))
    
delta_mean_d = [delta for mol_group in delta_mean_d for delta in mol_group]
delta_mean_a = [delta for mol_group in delta_mean_a for delta in mol_group]
delta_mean_t = [delta for mol_group in delta_mean_t for delta in mol_group]

delta_std_d = [delta for mol_group in delta_std_d for delta in mol_group]
delta_std_a = [delta for mol_group in delta_std_a for delta in mol_group]
delta_std_t = [delta for mol_group in delta_std_t for delta in mol_group]

print(np.mean(delta_mean_d), np.nanmean(delta_mean_a), np.mean(delta_mean_t))
print(np.std(delta_mean_d)/np.sqrt(len(delta_mean_d)), np.nanstd(delta_mean_a)/np.sqrt(len(delta_mean_a)), np.std(delta_mean_t)/np.sqrt(len(delta_mean_t)))
print(np.mean(delta_std_d), np.nanmean(delta_std_a), np.mean(delta_std_t))
print(np.std(delta_std_d)/np.sqrt(len(delta_std_d)), np.nanstd(delta_std_a)/np.sqrt(len(delta_std_a)), np.std(delta_std_t)/np.sqrt(len(delta_std_t)))
