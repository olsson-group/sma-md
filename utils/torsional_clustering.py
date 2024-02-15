import numpy as np
import matplotlib.pyplot as plt
import torch

from rdkit.Geometry import Point3D
from rdkit.Chem.rdMolAlign import  GetBestRMS

from scipy.signal import find_peaks, peak_prominences
from scipy.special import i0

from rdkit import Chem
from utils.z_matrix import deconstruct_z_matrix_batch

def vonmises_kde(data, kappa, n_bins=100):
    """
    Computes the kernel density estimate (KDE) using the von Mises distribution.

    Parameters:
    - data: numpy.ndarray
        The input data for which the KDE is computed.
    - kappa: float
        The concentration parameter of the von Mises distribution.
    - n_bins: int, optional
        The number of bins used for the histogram representation of the KDE. Default is 100.

    Returns:
    - bins: numpy.ndarray
        The bin edges of the histogram.
    - kde: numpy.ndarray
        The kernel density estimate values corresponding to each bin.
    """
    
    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= len(data)
    return bins, kde

def angular_distance(x1, x2):
    """
    Calculate the angular distance between two angles.
    
    Parameters:
        x1 (float): The first angle in radians.
        x2 (float): The second angle in radians.
    
    Returns:
        float: The angular distance between x1 and x2.
    """
    dist = np.abs(x1-x2)
    dist[dist>np.pi] = 2*np.pi-dist[dist>np.pi]
    return dist

def signed_angular_distance(x1, x2):
    """
    Calculates the signed angular distance between two angles.
    
    Parameters:
        x1 (float): The first angle in radians.
        x2 (float): The second angle in radians.
    
    Returns:
        float: The signed angular distance between x1 and x2.
    """
    dif = x1-x2
    dif[np.abs(dif)>np.pi] = -np.sign(dif[np.abs(dif)>np.pi])*(2*np.pi-np.abs(dif[np.abs(dif)>np.pi]))
    return dif

def get_peaks_torsion(data, n_bins=100, kappa = 100, threshold = 0.02, noise_threshold = 0.07, plot=False, plot_axes = None):
    """
    Gets peaks of torsional data based on the von Mises kernel density estimation.

    Args:
        data (array-like): Input data for clustering.
        n_bins (int, optional): Number of bins for the kernel density estimation. Default is 100.
        kappa (float, optional): Concentration parameter for the von Mises distribution. Default is 100.
        threshold (float, optional): Threshold value for peak prominence. Default is 0.02.
        noise_threshold (float, optional): Threshold value for considering noise. Default is 0.07.
        plot (bool, optional): Whether to plot the histogram and KDE. Default is False.
        plot_axes (matplotlib.axes.Axes, optional): Axes object for plotting. Default is None.

    Returns:
        array-like: Peaks of the clustered torsional data.

    """
    x, kde = vonmises_kde(data, kappa, n_bins=n_bins)
    range = np.max(kde)-np.min(kde)
    if range > noise_threshold:
        augmented_data = np.append(kde, [kde,kde])
        augmented_indexes = find_peaks(augmented_data)[0]
        indexes_center = augmented_indexes[np.where( (augmented_indexes>n_bins-1) & (augmented_indexes<2*n_bins))[0]]
        indexes = indexes_center-n_bins
        peaks = x[indexes]
        prominences, left_indexes, right_indexes = peak_prominences(augmented_data, indexes_center)
        left_min = kde[left_indexes%n_bins]
        right_min = kde[right_indexes%n_bins]
        mins = np.max(np.append(np.expand_dims(left_min, axis=-1), np.expand_dims(right_min, axis =-1), axis = 1), axis = 1)
    else:
        peaks = np.array([0])
        prominences = np.array([threshold+0.1])
        indexes = np.array([int(n_bins/2)])
    if plot:
        if not plot_axes:
            fig, plot_axes = plt.subplots()
        plot_axes.hist(data, n_bins, density=True)
        plt.xlim([-np.pi, np.pi])
        plot_axes.plot(x, kde, color="orange")
        if np.any(prominences > threshold):
            plot_axes.scatter(peaks[np.where(prominences>threshold)], kde[indexes[np.where(prominences>threshold)]], color="red", s = 120, marker="x")
    peaks = peaks[np.where(prominences>threshold)]

    return peaks

def get_peaks_molecule(z_matrices, rb_atoms, n_bins=100, kappa=100, threshold=0.02, plot=False, all_torsions=False):
    """
    Get peaks of torsional angles for a molecule.

    Parameters:
    - z_matrices (numpy.ndarray): Array of shape (n_samples, n_atoms, 3) representing the molecular structure.
    - rb_atoms (list): List of atom indices for which torsional angles will be clustered.
    - n_bins (int): Number of bins for histogram-based clustering (default: 100).
    - kappa (float): Kappa parameter for von Mises distribution clustering (default: 100).
    - threshold (float): Threshold for peak detection in clustering (default: 0.02).
    - plot (bool): Whether to plot KDEs and data with peaks (default: False).
    - all_torsions (bool): Whether to cluster all torsional angles or only those specified in rb_atoms (default: False).

    Returns:
    - peaks (list): List of peak values for each torsional angle cluster.
    """
        
    if plot:
        if all_torsions:
            fig, axs = plt.subplots(1, z_matrices.shape[1]-2, figsize=(40, 10.))
            for i_plot, i_torsion in enumerate(range(2,z_matrices.shape[1])):
                axs[i_plot].set(title=str(i_torsion+1))
        else:
            fig, axs = plt.subplots(1, len(rb_atoms), figsize=(40, 10.))
            for i_plot, i_rb in enumerate(rb_atoms):
                axs[i_plot].set(title=str(i_rb))
    else:
        axs = [None for _ in rb_atoms]

    peaks = []
    if all_torsions:
        for i_plot, i_torsion in enumerate(range(2,z_matrices.shape[1])):
            if plot:
                plot_axes = axs[i_plot]
            else:
                plot_axes = None
            peaks.append(get_peaks_torsion(z_matrices[:,i_torsion,2], plot=plot, plot_axes=plot_axes, n_bins=n_bins, kappa=kappa, threshold=threshold))
    else:
        for i_plot, i_torsion_atom in enumerate(rb_atoms):
            if plot:
                plot_axes = axs[i_plot]
            else:
                plot_axes = None
            peaks.append(get_peaks_torsion(z_matrices[:,i_torsion_atom-1,2], plot=plot, plot_axes=plot_axes, n_bins=n_bins, kappa=kappa, threshold=threshold))

    return peaks

def get_conformers(z_matrices, rb_atoms, mol_dic, gen_z_matrices=None, n_bins=100, kappa = 100, threshold = 0.02, delta_rmsd = 0.05):
    """
    Generate conformers based on torsional clustering.

    Parameters:
    z_matrices (numpy.ndarray): Array of shape (n, m, 3) representing the input z-matrices.
    rb_atoms (list): List of atom indices involved in the rigid body.
    mol_dic (dict): Dictionary containing molecular information.
    gen_z_matrices (numpy.ndarray, optional): Array of shape (n, m, 3) representing the generated z-matrices. Defaults to None.
    n_bins (int, optional): Number of bins for torsion histograms. Defaults to 100.
    kappa (int, optional): Kappa parameter for clustering. Defaults to 100.
    threshold (float, optional): Threshold value for peak detection. Defaults to 0.02.
    delta_rmsd (float, optional): Threshold value for RMSD comparison. Defaults to 0.05.

    Returns:
    numpy.ndarray: Array of shape (k, m, 3) representing the generated conformers.
    list: List of atom indices involved in the conformers.
    """
    #get peaks for torsions histograms and remove symmetries in terminal atoms
    if type(gen_z_matrices) != type(None):
        peaks = get_peaks_molecule(gen_z_matrices, rb_atoms, n_bins=n_bins, kappa=kappa ,threshold=threshold, all_torsions = True)
        torsion_clusters, atom_conformers = clean_peaks(mol_dic, peaks)
    else:
        gen_z_matrices = z_matrices.copy()
        peaks = get_peaks_molecule(z_matrices, rb_atoms, n_bins=n_bins, kappa=kappa, threshold=threshold, all_torsions = True)
        torsion_clusters, atom_conformers = clean_peaks(mol_dic, peaks)
    #get pertenence to clusters in torsional space
    defining_peaks = torsion_clusters
    pertenence = get_pertenenece(z_matrices, torsion_clusters, atom_conformers)

    #check what torsion sequences appear in data
    actual_conformers = 0
    found_seq = []
    for seq in pertenence:
        if seq.tolist() not in found_seq:
            found_seq.append(seq.tolist())
            actual_conformers+=1
    found_seq.sort()
    for i_torsion in range(len(found_seq[0])):
        peaks_in_torsions = [found_seq[i_seq][i_torsion] for i_seq in range(len(found_seq))]
        if len(set(peaks_in_torsions)) == 1:
            atom_conformers.remove(atom_conformers[i_torsion])
    #construct z-matrices for them
    base_z_matrix = np.mean(gen_z_matrices, axis=0)
    conformers_z_matrices = np.zeros((len(found_seq), len(base_z_matrix), 3 )) 
    for i_seq, seq in enumerate(found_seq):
        conformers_z_matrices[i_seq] = base_z_matrix.copy()
        conf_torsions = np.array([ defining_peaks[i_peak][seq[i_peak]] for i_peak in range(len(defining_peaks)) ])
        conformers_z_matrices[i_seq, [conf_atom-1 for conf_atom in atom_conformers], 2] = conf_torsions.copy()
    
    conformers = deconstruct_z_matrix_batch(torch.tensor(conformers_z_matrices), mol_dic["ref_atoms"])
    conformers = conformers.detach().numpy().astype("double")
    #check for inversions in conformers or conformers that are too similar
    conformers_to_keep = []
    mol_1 = mol_dic["rdkit_mol"]
    mol_2 = Chem.Mol(mol_1)
    for i_conf, conf_i in enumerate(conformers):
        conf_1 = mol_1.GetConformer()
        for i in range(mol_1.GetNumAtoms()):
            x,y,z = conf_i[i]
            conf_1.SetAtomPosition(i,Point3D(x,y,z))
        rmsd = []
        for j_conf, conf_j in enumerate(conformers[i_conf+1:]):    
            conf_2 = mol_2.GetConformer()
            for i in range(mol_2.GetNumAtoms()):
                x,y,z = conf_j[i]
                conf_2.SetAtomPosition(i,Point3D(x,y,z))
            
            rmsd.append(GetBestRMS(mol_1, mol_2))
        rmsd = np.array(rmsd)

        rmsd_inverse = []
        for j_conf, conf_j in enumerate(conformers[i_conf+1:]):    
            conf_2 = mol_2.GetConformer()
            for i in range(mol_2.GetNumAtoms()):
                x,y,z = -conf_j[i]
                conf_2.SetAtomPosition(i,Point3D(x,y,z))
            
            rmsd_inverse.append(GetBestRMS(mol_1, mol_2))
        rmsd_inverse = np.array(rmsd_inverse)
        inverse_indexes = np.where(rmsd_inverse<rmsd)
        rmsd[inverse_indexes] = rmsd_inverse[inverse_indexes]

        if not np.any(rmsd<delta_rmsd):
            conformers_to_keep.append(i_conf)
        
    return conformers[conformers_to_keep], atom_conformers


def clean_peaks(mol_dic, peaks):
    """
    Discard symmetries in terminal atoms. Assumes peaks has been computed with all_torsions=True.

    Args:
        mol_dic (dict): A dictionary containing molecular information.
        peaks (list): A list of peaks.

    Returns:
        tuple: A tuple containing torsion conformers and atoms indexes.
    """
    torsion_atoms = np.arange(3, len(mol_dic["atoms"]))
    rb_atoms = [group[2] for group in mol_dic["groups"][1:]]
    
    no_count_atoms = []
    for i_group, group in enumerate(mol_dic["groups"]):
        if len(group[2:])>1:
            if np.all(np.sum(Chem.GetAdjacencyMatrix(mol_dic["rdkit_mol"])[group[2:]], axis=1)==1):
                if np.all(mol_dic["atoms"][group[2:]]==mol_dic["atoms"][group[2]]):
                        if i_group>0:
                            if len(peaks[group[2]-3]) <= len(group)-2:
                                no_count_atoms.append(group[2])
                        else:
                            if len(peaks[mol_dic["groups"][1][2]-3]) <= len(group)-2:
                                no_count_atoms.append(mol_dic["groups"][1][2])
    no_count_atoms = list(set(no_count_atoms))
    no_count_peaks = [no_count_atom-3 for no_count_atom in no_count_atoms]

    torsion_conformers = []
    atoms_indexes = []
    
    for i_peak, peak in enumerate(peaks):
        if i_peak not in no_count_peaks and len(peak)>1:
            torsion_conformers.append(peak)
            atoms_indexes.append(torsion_atoms[i_peak])

    return torsion_conformers, atoms_indexes

def get_pertenenece(z_matrices, torsion_clusters, atom_conformers):
    """
    Calculates the pertenence matrix for torsional clustering.

    Args:
        z_matrices (numpy.ndarray): Array of shape (N, M, 3) representing the atomic coordinates.
        torsion_clusters (list): List of lists containing the torsion clusters for each atom.
        atom_conformers (list): List of atom indices for which the pertenence matrix is calculated.

    Returns:
        numpy.ndarray: Array of shape (N, len(atom_conformers)) representing the pertenence matrix.
    """
    pertenence = np.zeros((len(z_matrices), len(atom_conformers)), dtype="int")
    for i_atom, atom in enumerate(atom_conformers):
        if len(torsion_clusters[i_atom]) > 1:
            dist = np.zeros((len(z_matrices), len(torsion_clusters[i_atom])))
            for i_peak, peak in enumerate(torsion_clusters[i_atom]):        
                peaks_batch = np.array([peak for _ in range(len(z_matrices))])
                dist[:, i_peak] = angular_distance(np.remainder(z_matrices[:, atom-1, 2]+2*np.pi, 2*np.pi), np.remainder(peaks_batch[i_atom]+2*np.pi, 2*np.pi))
                pertenence[:, i_atom] = np.argmin(dist, axis=1)

    return pertenence