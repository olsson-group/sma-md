import numpy as np
from tqdm import tqdm
from scipy.special import rel_entr

test_set_indices = np.load("datasets/mdqm9-nc/splits/test_indices.npy")[:100]
model_name = "dls_r_4"
max_energy = 1e5

energy_kls = []
energy_mean_diff = []

for i_mol in tqdm(test_set_indices):
    i_mol = int(i_mol)
    
    other_energies_filename = "sampled_data/energies/md_tuned/dls_r_4/energies_sampled_{:0=5}_1_ns_rw.npy".format(i_mol)
    
    other_energies = np.load(other_energies_filename)
    other_indices = other_energies<max_energy
    other_energies = other_energies[other_indices]
    
    target_energies = np.load("sampled_data/energies/re/{:0=5}.npy".format(i_mol))
    
    density_re, chosen_bins = np.histogram(target_energies, bins = 100, density = True)
    density_other, _ = np.histogram(other_energies, bins = chosen_bins, density = True)
    
    kl_div = 0.5*rel_entr(density_re, density_other) + 0.5*rel_entr(density_other, density_re)
    kl_div = np.sum(kl_div[kl_div != np.inf])
    energy_kls.append(kl_div)
    energy_mean_diff.append(np.abs(np.mean(other_energies)-np.mean(target_energies)))
    
print(np.mean(energy_kls), "pm", np.std(energy_kls)/np.sqrt(len(energy_kls)))
print(np.mean(energy_mean_diff), "pm", np.std(energy_mean_diff)/np.sqrt(len(energy_mean_diff)))
