import os
import numpy as np
import sys
from openmm import unit
from tqdm import tqdm

from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
import openmm
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField
from numpy.random import choice

from parameters import params
from utils.mdqm9_loader import MDQM9EvalDataset

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

p = Struct(**params)

dataset = MDQM9EvalDataset(p.sdf_path, p.hdf5_path)

if not os.path.isdir("sampled_data/energies/"):
    os.mkdir("sampled_data/energies/")

mol_indices = np.load(sys.argv[1])
mode = sys.argv[2]
model_name = sys.argv[3]

directories = ["sampled_data", "sampled_data/energies/", "sampled_data/energies/contributions/",
               "sampled_data/energies/"+mode, "sampled_data/energies/contributions/"+mode]
for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)

for i_mol in mol_indices:
    print(f"Evaluating energy of mol {i_mol} ...", end = ' ', flush=True)
    mol_dic = dataset[int(i_mol)]
    
    if mode == "gen_model":
        confs = np.load("sampled_data/conformations/gen_model/"+model_name+"/{:0=5}.npy".format(i_mol))
    elif mode == "md_tuned":
        filename = "{:0=5}".format(i_mol)+"_1_ns_rw"
        confs = np.load("sampled_data/conformations/md_tuned/"+model_name+"/"+filename+".npy")
    elif mode == "md":
        confs = mol_dic["conformations"]
    elif mode == "re":
        confs = mol_dic["re_conformations"]
    elif mode == "md_rt":
        confs = mol_dic["mdrt_conformations"]
    else:
        print("Mode not recognized.", flush=True)
        break

    mol_dic = dataset[i_mol]

    rd_mol = mol_dic["rdkit_mol"]
    partial_charges = mol_dic["partial_charges"]

    off_mol = Molecule.from_rdkit(rd_mol)
    off_mol.partial_charges = unit.Quantity(value = np.array(partial_charges), unit = unit.elementary_charge)
    gaff = GAFFTemplateGenerator(molecules=off_mol)

    topology = Topology.from_molecules(off_mol).to_openmm()
    forcefield = ForceField('amber/protein.ff14SB.xml', "implicit/gbn2.xml")
    forcefield.registerTemplateGenerator(gaff.generator) 
    system = forcefield.createSystem(topology)

    forces = { force.__class__.__name__ : force for force in system.getForces() }
    nbforce = forces['NonbondedForce']

    energies = []
    energy_contributions = np.zeros((len(confs), len(system.getForces())))

    for i_conf, conf in enumerate(confs):
        integrator = openmm.LangevinIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
        context = openmm.Context(system, integrator)
        context.setPositions(conf) #Positions must be given in nm!
        state = context.getState(getEnergy=True)
        #take these two lines out
        kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        energy_factor = 1. / (integrator.getTemperature() * kB_NA) #0.4 mol/kj
        energy = state.getPotentialEnergy() * energy_factor
        energies.append(energy)
        for i_contribution, f in enumerate(system.getForces()):
            state = context.getState(getEnergy=True, groups={i_contribution})
            energy_contributions[i_conf, i_contribution] = state.getPotentialEnergy()*energy_factor
            
    print("Done!", flush=True)

    if not os.path.isdir("sampled_data/energies/"):
        os.mkdir("sampled_data/energies/")

    if mode == "gen_model":        
        energies_filename = "sampled_data/energies/gen_model/"+model_name+"/energies_sampled_{:0=5}".format(i_mol)+"_water.npy"
        contributions_filename = "sampled_data/energies/contributions/gen_model/"+model_name+"/contributions_sampled_{:0=5}".format(i_mol)+"_water.npy"
        
        np.save(energies_filename, energies)
        np.save(contributions_filename, energy_contributions)
    else:
        directories = ["sampled_data/energies/"+mode, "sampled_data/energies/contributions/"+mode]
        for dir in directories:
            if not os.path.isdir(dir):
                os.mkdir(dir)
                
        energies_filename = "sampled_data/energies/"+mode+"/energies_sampled_{:0=5}".format(i_mol)+"_water.npy"
        contributions_filename = "sampled_data/energies/contributions/"+mode+"/contributions_sampled_{:0=5}".format(i_mol)+"_water.npy"
        np.save(energies_filename, energies)
        np.save(contributions_filename, energy_contributions)