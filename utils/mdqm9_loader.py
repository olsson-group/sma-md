import h5py
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset

class MDQM9Dataset(Dataset):
    def __init__(self, sdf_path, hdf5_path):
        super().__init__()
        self.sdf_path = sdf_path
        self.hdf5_path = hdf5_path
        
        self.hdf5_dataset = h5py.File(hdf5_path,'r')
        self.mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs = False, sanitize = True)
        
    def __len__(self):
        # Number of molecules in the dataset
        return len(self.mol_supplier)

    def __getitem__(self, idx):
        rdkit_mol = self.mol_supplier[int(idx)]
        formated_idx = "{:0>5d}".format(idx)
        atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["atoms"])
        heavy_atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["heavy_atoms"])
        time_lag = np.array(self.hdf5_dataset[formated_idx]["data"]["time_lag"])
        conformations = np.array(self.hdf5_dataset[formated_idx]["trajectories"]["md_0"])
        partial_charges = np.array(self.hdf5_dataset[formated_idx]["data"]["partial_charges"])
        ref_atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["ref_atoms"])
        groups = list(self.hdf5_dataset[formated_idx]["data"]["groups"][:])
        
        return {"rdkit_mol": rdkit_mol, "atoms": atoms, "heavy_atoms": heavy_atoms, "time_lag": time_lag, "partial_charges": partial_charges,
                "groups": groups, "ref_atoms": ref_atoms, "conformations": conformations,  "idx": idx}
        
class MDQM9EvalDataset(Dataset):
    def __init__(self, sdf_path, hdf5_path):
        super().__init__()
        self.sdf_path = sdf_path
        self.hdf5_path = hdf5_path
        
        self.hdf5_dataset = h5py.File(hdf5_path,'r')
        self.mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs = False, sanitize = True)
        
    def __len__(self):
        # Number of molecules in the dataset
        return len(self.mol_supplier)

    def __getitem__(self, idx):
        rdkit_mol = self.mol_supplier[int(idx)]
        formated_idx = "{:0>5d}".format(idx)
        atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["atoms"])
        heavy_atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["heavy_atoms"])
        time_lag = np.array(self.hdf5_dataset[formated_idx]["data"]["time_lag"])
        partial_charges = np.array(self.hdf5_dataset[formated_idx]["data"]["partial_charges"])
        ref_atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["ref_atoms"])
        groups = list(self.hdf5_dataset[formated_idx]["data"]["groups"][:])
        
        conformations = np.array(self.hdf5_dataset[formated_idx]["trajectories"]["md_0"])
        labels = self.hdf5_dataset[formated_idx]["trajectories"].keys()
        if "mdrt_0" in labels:
            mdrt_conformations = np.array(self.hdf5_dataset[formated_idx]["trajectories"]["mdrt_0"])
        else:
            mdrt_conformations = None
        if "re_0" in labels:
            re_conformations = np.array(self.hdf5_dataset[formated_idx]["trajectories"]["re_0"])
        else:    
            re_conformations = None
        
        return {"rdkit_mol": rdkit_mol, "atoms": atoms, "heavy_atoms": heavy_atoms, "time_lag": time_lag, "partial_charges": partial_charges,
                "groups": groups, "ref_atoms": ref_atoms, "conformations": conformations, "mdrt_conformations": mdrt_conformations, 
                "re_conformations": re_conformations,  "idx": idx}
                

class MinimalMDQM9Dataset(Dataset):
    def __init__(self, sdf_path, hdf5_path):
        super().__init__()
        self.sdf_path = sdf_path
        self.hdf5_path = hdf5_path
        
        self.hdf5_dataset = h5py.File(hdf5_path,'r')
        self.mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs = False, sanitize = True)
        
    def __len__(self):
        # Number of molecules in the dataset
        return len(self.mol_supplier)

    def __getitem__(self, idx):
        rdkit_mol = self.mol_supplier[idx]
        formated_idx = "{:0>5d}".format(idx)
        conformations = np.array(self.hdf5_dataset[formated_idx]["trajectories"]["md_0"])
        partial_charges = np.array(self.hdf5_dataset[formated_idx]["data"]["partial_charges"])
        
        return {"rdkit_mol": rdkit_mol, "conformations": conformations, "partial_charges": partial_charges, "idx": idx}
    
class TrainingMDQM9Dataset(Dataset):
    def __init__(self, sdf_path, hdf5_path):
        super().__init__()
        self.sdf_path = sdf_path
        self.hdf5_path = hdf5_path
        
        self.hdf5_dataset = h5py.File(hdf5_path,'r')
        self.mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs = False, sanitize = True)
        
    def __len__(self):
        # Number of molecules in the dataset
        return len(self.mol_supplier)

    def __getitem__(self, idx):
        rdkit_mol = self.mol_supplier[idx]
        formated_idx = "{:0>5d}".format(idx)
        conformations = np.array(self.hdf5_dataset[formated_idx]["trajectories"]["md_0"])
        partial_charges = np.array(self.hdf5_dataset[formated_idx]["data"]["partial_charges"])
        ref_atoms = np.array(self.hdf5_dataset[formated_idx]["data"]["ref_atoms"])
        groups = list(self.hdf5_dataset[formated_idx]["data"]["groups"][:])
        
        return {"rdkit_mol": rdkit_mol, "conformations": conformations, "partial_charges": partial_charges,
                "groups": groups, "ref_atoms": ref_atoms,  "idx": idx}