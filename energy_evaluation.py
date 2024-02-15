import time
import os
import sys

import numpy as np
from simtk import unit
from tqdm import tqdm
import mdtraj as md

from utils.openmm_replicated_systems import OMMTReplicas_replicated
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField

from parameters import params
from utils.mdqm9_loader import MDQM9EvalDataset

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

p = Struct(**params)

dataset = MDQM9EvalDataset(p.sdf_path, p.hdf5_path)

save_total_time = True    

mol_indices = np.load(sys.argv[1])
mode = sys.argv[2]
model_name = sys.argv[3]

directories = ["sampled_data", "sampled_data/energies/", "sampled_data/energies/contributions/",
               "sampled_data/energies/"+mode, "sampled_data/energies/contributions/"+mode]
for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)
        
if save_total_time:
    if not os.path.isdir("sampled_data/performance/time_profile/energy_evaluation/"):
        os.mkdir("sampled_data/performance/time_profile/energy_evaluation")

for i_mol in mol_indices:
    i_mol = int(i_mol)
    mol_dic = dataset[i_mol]
    print(f"Evaluating energy of mol {i_mol} ...", end = ' ', flush=True)

    if mode == "gen_model":
        confs = np.load("sampled_data/conformations/gen_model/"+model_name+"/{:0=5}.npy".format(i_mol))
    elif mode == "md":
        confs = mol_dic["conformations"]
    elif mode == "md_tuned":
        filename = "{:0=5}".format(i_mol)+"_1_ns_rw"
        confs = np.load("sampled_data/conformations/md_tuned/"+model_name+"/"+filename+".npy")  
    elif mode == "re":
        confs = mol_dic["re_conformations"]
    elif mode == "md_rt":
        confs = mol_dic["mdrt_conformations"]
    else:
        print("Mode not recognized.", flush=True)
        break

    rd_mol = mol_dic["rdkit_mol"]
    partial_charges = mol_dic["partial_charges"]

    off_mol = Molecule.from_rdkit(rd_mol)
    off_mol.partial_charges = unit.Quantity(value = np.array(partial_charges), unit = unit.elementary_charge)
    gaff = GAFFTemplateGenerator(molecules=off_mol)
    
    topology = Topology.from_molecules(off_mol).to_openmm()
    forcefield = ForceField('amber/protein.ff14SB.xml')
    forcefield.registerTemplateGenerator(gaff.generator)
    system = forcefield.createSystem(topology)
    
    forces = { force.__class__.__name__ : force for force in system.getForces() }

    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)

    t0 = time.time()

    temps = [300. for _ in range(len(confs))] # in K
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": 1.0}
    kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    energy_factor = 1. / ( unit.Quantity(value = 300, unit = unit.kelvin)* kB_NA)

    simu = OMMTReplicas_replicated(system, temps, integrator_params=integrator_params, platform="CUDA")
    simu.set_positions_all(confs)

    t1 = time.time()
    #print("Setting positions:", t1-t0, "s.")

    energies = np.zeros((len(confs)))
    energy_contributions = np.zeros((len(confs), len(system.getForces())))

    for i in range(len(confs)):
        contributions = simu.get_energy_contributions(i)
        energy_contributions[i, :] = contributions*energy_factor
        
    energies = np.sum(energy_contributions, axis=1)

    t2 = time.time()

    #print("Evaluating energies:", t2-t1, "s.")
    print(t2-t0, "s.")

    
    if mode == "gen_model":
        if not os.path.isdir("sampled_data/energies/gen_model/"+model_name):
            os.mkdir("sampled_data/energies/gen_model/"+model_name)
        if not os.path.isdir("sampled_data/energies/contributions/gen_model/"+model_name):
            os.mkdir("sampled_data/energies/contributions/gen_model/"+model_name)
            
        energies_filename = "sampled_data/energies/gen_model/"+model_name+"/{:0=5}".format(i_mol)+".npy"
        contributions_filename = "sampled_data/energies/contributions/gen_model/"+model_name+"/{:0=5}".format(i_mol)+".npy"
        
        np.save(energies_filename, energies)
        np.save(contributions_filename, energy_contributions)
        
        if save_total_time:
            np.save("sampled_data/performance/time_profile/energy_evaluation/ee_{:0=5}_".format(i_mol)+str(int(len(energies)/1000))+"k.npy", [t1-t0, t2-t1])
    else:
        directories = ["sampled_data/energies/"+mode, "sampled_data/energies/contributions/"+mode]
        for dir in directories:
            if not os.path.isdir(dir):
                os.mkdir(dir)
        
        energies_filename = f"sampled_data/energies/{mode}/"+"{:0=5}".format(i_mol)+".npy"
        contributions_filename = f"sampled_data/energies/contributions/{mode}/"+"/{:0=5}".format(i_mol)+".npy"
        np.save(energies_filename, energies)
        np.save(contributions_filename, energy_contributions)

