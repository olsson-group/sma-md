import time
import os
import sys

import numpy as np
from openmm import unit
from tqdm import tqdm
import torch
from numpy.random import choice
from scipy.stats import iqr

from utils.openmm_replicated_systems import OMMTReplicas_replicated
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField
#import openmm

from parameters import params
from utils.mdqm9_loader import MDQM9Dataset

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

p = Struct(**params)

dataset = MDQM9Dataset(p.sdf_path, p.hdf5_path)

#platform = openmm.Platform.getPlatformByName('CPU')

mol_indices = np.load(sys.argv[1])
model_name = sys.argv[2]

integration_time = 1.0 #fs
ft_steps = int(1e6) # (1e6 = 1 ns)
temperature = 300 #K
re_weighting = True # assumes energies have being already computed
evaluate_energy = True #after simulation
filter_conformers = True #ignore conformers with very high energy
save_total_time = True

directories = ["sampled_data/conformations/md_tuned", "sampled_data/conformations/md_tuned/"+ model_name,
               "sampled_data/energies/md_tuned", "sampled_data/energies/md_tuned/"+ model_name,
               "sampled_data/energies/contributions/md_tuned", "sampled_data/energies/contributions/md_tuned/"+ model_name,
               "sampled_data/performance/time_profile/md_ft", "sampled_data/performance/time_profile/md_ft/"+ model_name]

for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)  
    
for i_mol in mol_indices:
    i_mol = int(i_mol)
    print(f"Fine-tuning mol {i_mol} ...", end = " ")

    sampled_positions = np.load("sampled_data/conformations/gen_model/"+model_name+"/{:0=5}.npy".format(i_mol))

    if re_weighting:
        contributions = np.load("sampled_data/energies/contributions/gen_model/"+model_name+"/{:0=5}.npy".format(i_mol))
        torsional_nb_energies = torch.tensor(contributions[:,1]+contributions[:,2])
        torsional_logps = torch.tensor(np.load("sampled_data/log_prob/"+ model_name + "/{:0=5}.npy".format(i_mol)))#in cartesian coordinates
        if len(torsional_nb_energies) == torch.sum(torch.isnan(torsional_nb_energies)) :
            print("Failed to re-weight molecule {}, jumping to next one.".format(i_mol))
            continue
        
        torsional_nb_energies[torch.isnan(torsional_nb_energies)] = np.inf
        log_w = - torsional_nb_energies - torsional_logps
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        w = np.exp(log_w).detach().numpy()
        draw = choice(np.arange(len(torsional_nb_energies)), len(torsional_nb_energies),
                p=w)
        positions = sampled_positions[draw]
    else:
        positions = sampled_positions

    mol_dic = dataset[i_mol]
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

    temps = [temperature for _ in range(len(positions))] # in K
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": integration_time}
    kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    energy_factor = 1. / ( unit.Quantity(value = 300, unit = unit.kelvin)* kB_NA)

    simu = OMMTReplicas_replicated(system, temps, integrator_params=integrator_params, platform="CUDA")
    print(f"Setting positions...", end=" ")
    simu.set_positions_all(positions)

    t1 = time.time()
    print(t1-t0, " s.", end="\t")

    #mdft

    simu._contexts[0].setVelocitiesToTemperature(temperature*unit.kelvin)
    print(f"Simulating...", end=" ")
    simu.step(ft_steps)
    ft_positions = simu.get_positions_all()

    if re_weighting:
        filename = "{:0=5}".format(i_mol)+"_"+str(int(ft_steps/1e6))+"_ns_rw"
    else:
        filename = "{:0=5}".format(i_mol)+"_"+str(int(ft_steps/1e6))+"_ns"
    np.save("sampled_data/conformations/md_tuned/"+model_name+"/"+filename+".npy", ft_positions)
    t2 = time.time()
    print(t2-t1, " s.", end="\t")

    if evaluate_energy:
        energies = np.zeros((len(ft_positions)))
        energy_contributions = np.zeros((len(ft_positions), len(system.getForces())))
        print(f"Evaluating energy...", end=" ")

        for i in range(len(ft_positions)):
            contributions = simu.get_energy_contributions(i)
            energy_contributions[i, :] = contributions*energy_factor
            
        energies = np.sum(energy_contributions, axis=1)
        indices = np.where(energies<np.inf)
        energies = energies[indices]
        energy_contributions = energy_contributions[indices]
        ft_positions = ft_positions[indices]

        t3 = time.time()
        print(t3-t2, " s.", flush=True)
        
        if filter_conformers:
            median_energy = np.median(energies)
            iqr_energy = iqr(energies)
            to_keep = np.where(energies<median_energy+5*iqr_energy/2)
            print("Kept "+str(len(to_keep[0]))+" conformers.")
            
            final_positions = ft_positions[to_keep]
            final_energy_contributions = energy_contributions[to_keep]
            final_energies = np.sum(final_energy_contributions, axis=1)
            
            np.save("sampled_data/conformations/md_tuned/"+model_name+"/"+filename+".npy", final_positions)
            np.save("sampled_data/energies/md_tuned/"+model_name+"/energies_sampled_"+filename+".npy", final_energies)
            np.save("sampled_data/energies/contributions/md_tuned/"+model_name+"/contributions_md_tuned_"+filename+".npy", final_energy_contributions)
            
        if save_total_time:
            np.save("sampled_data/performance/time_profile/md_ft/md_"+filename+".npy", [t1-t0, t2-t1])


