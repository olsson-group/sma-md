import os

import torch
import numpy as np
from tqdm import tqdm
import h5py
import openmm
from openmm import unit
from openmm.app import ForceField

from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
from openmmforcefields.generators import GAFFTemplateGenerator

from utils.mdqm9_loader import MDQM9Dataset
from utils.z_matrix import construct_z_matrix_batch, deconstruct_z_matrix_batch

sdf_path = "datasets/mdqm9-nc/mdqm9-nc.sdf"       
hdf5_path = "datasets/mdqm9-nc/mdqm9-nc.hdf5"

dataset = MDQM9Dataset(sdf_path, hdf5_path)

if not os.path.isdir("datasets/"):
    os.mkdir("datasets/")
    
if not os.path.isdir("datasets/auxiliary_datasets/"):
    os.mkdir("datasets/auxiliary_datasets/")
    
f = h5py.File('datasets/auxiliary_datasets/aux-mdqm9-nc.hdf5','w')

for idx in tqdm(range(len(dataset))):

    mol_dic = dataset[idx]

    rd_mol = mol_dic["rdkit_mol"]
    
    partial_charges = mol_dic["partial_charges"]
    groups = mol_dic["groups"]
    ref_atoms = mol_dic["ref_atoms"]
    
    md_z_matrices = construct_z_matrix_batch(torch.tensor(mol_dic["conformations"]), ref_atoms)

    off_mol = Molecule.from_rdkit(rd_mol)
    off_mol.partial_charges = unit.Quantity(value = np.array(partial_charges), unit = unit.elementary_charge)
    gaff = GAFFTemplateGenerator(molecules=off_mol)
    forcefield = ForceField('amber/protein.ff14SB.xml')
    forcefield.registerTemplateGenerator(gaff.generator)
    topology = Topology.from_molecules(off_mol).to_openmm()
    try:     
        system = forcefield.createSystem(topology)
    except Exception as e:
        print(f"Failed to create system of molecule {idx}.", flush=True)
        print(e, flush=True)
        continue
    
    forces = { force.__class__.__name__ : force for force in system.getForces() }
        
    integration_time = 1 #fs
    temperature = 300 #K

    integrator = openmm.LangevinIntegrator(temperature*unit.kelvin, 1.0/unit.picoseconds, integration_time*unit.femtoseconds)
    integrator.setConstraintTolerance(0.00001)

    platform = openmm.Platform.getPlatformByName('CPU')
    simulation = openmm.app.Simulation(topology, system, integrator, platform)

    d_z_matrix = torch.zeros((len(ref_atoms)-1,3))
    for force_idx in range(system.getNumForces()):
            force = system.getForce(force_idx)

            #get equilibrium bond lengths
            if isinstance(force, openmm.HarmonicBondForce):
                for bond_idx in range(force.getNumBonds()):
                    atom1, atom2, length, k = force.getBondParameters(bond_idx)
                    if atom2>atom1 and ref_atoms[atom2][0] == atom1: 
                        d_z_matrix[atom2-1,0] = length.value_in_unit(unit.nanometer)
                    elif atom1>atom2 and ref_atoms[atom1][0] == atom2: 
                        d_z_matrix[atom1-1,0] = length.value_in_unit(unit.nanometer)
                    else:
                        print(f"Warning, bond {atom1}-{atom2} was not associated with with entry in z-matrix.")
                        
            #get equilibrium angles
            if isinstance(force, openmm.HarmonicAngleForce):
                for angle_idx in range(force.getNumAngles()):
                    atom1, atom2, atom3, angle, k = force.getAngleParameters(angle_idx)
                    if atom3>atom1 and ref_atoms[atom3][0] == atom2 and ref_atoms[atom3][1] == atom1:
                        angle = angle.value_in_unit(unit.radians)
                        if np.abs(angle-np.pi) > 1e-5: 
                            d_z_matrix[atom3-1,1] = angle
                        else:
                            d_z_matrix[atom3-1,1] = angle - 1e-5
                    elif atom3<atom1 and ref_atoms[atom1][0] == atom2 and ref_atoms[atom1][1] == atom3: 
                        angle = angle.value_in_unit(unit.radians)
                        if np.abs(angle-np.pi) > 1e-5: 
                            d_z_matrix[atom1-1,1] = angle
                        else:
                            d_z_matrix[atom1-1,1] = angle - 1e-5

    d_z_matrices = d_z_matrix.repeat(len(md_z_matrices),1,1)
    rotable_bonds_z_entries = [group[2]-1 for group in groups[1:]]
    
    for group in groups:
        #check hybridization and assign dihedral angle
        if len(group) == 4:  
            atom = rd_mol.GetAtomWithIdx(int(group[0]))
            hybridization = str(atom.GetHybridization())
            if hybridization == "SP2":
                d_z_matrices[:, group[3]-1, 2] = torch.pi
            elif hybridization == "SP3":
                d_z_matrices[:, group[3]-1, 2] = torch.sign(md_z_matrices[:,group[3]-1, 2])*2*torch.pi/3
            else:
                print("Warning, hybridization not recognized.")
        if len(group) == 5:
            #keep the same isomers.
            d_z_matrices[:, group[3]-1, 2] = torch.sign(md_z_matrices[:,group[3]-1, 2])*2*torch.pi/3
            d_z_matrices[:, group[4]-1, 2] = torch.sign(md_z_matrices[:,group[4]-1, 2])*2*torch.pi/3
              
        d_z_matrices[:, rotable_bonds_z_entries, 2] = md_z_matrices[:, rotable_bonds_z_entries, 2].clone()

    positions = deconstruct_z_matrix_batch(d_z_matrices, ref_atoms)
    
    #hdf5 dataset
    
    formated_idx = "{:0>5d}".format(idx)
    grp = f.create_group("{:0>5d}".format(idx))
    data_subgrp = grp.create_group("data")

    atoms = mol_dic["atoms"]
    n_heavy_atoms = mol_dic["heavy_atoms"]
    time_lag = mol_dic["time_lag"]
    
    atoms_ds = data_subgrp.create_dataset("atoms", data=atoms)
    data_subgrp.create_dataset("heavy_atoms", data=n_heavy_atoms) 
    data_subgrp.create_dataset("time_lag", data=time_lag) #ps
    partial_charges_ds = data_subgrp.create_dataset("partial_charges", data=partial_charges)
    data_subgrp.create_dataset("groups", shape=len(groups), dtype=h5py.vlen_dtype(np.dtype('int32')))
    for i, group in enumerate(groups):
        data_subgrp["groups"][i] = group
    data_subgrp.create_dataset('ref_atoms', data=ref_atoms)
    
    traj_subgrp = grp.create_group("trajectories")

    cc = positions.numpy()

    cc_ds = traj_subgrp.create_dataset("md_0", data=cc)
    
f.close()