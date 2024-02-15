from utils.mol_geometry import compute_torsion, compute_angle, compute_distance, ic_to_xyz, ic_to_xyz_test
import torch
import numpy as np

def construct_z_matrix(X, ref_atoms, placing_order=None):
    '''
    Given a torch.tensor array, whose rows are the cartesian coordinates of a molecule, generates a Z-matrix computing internal coordinates.

    Arguments:

    - X (torch.tensor, size: (n_atoms, 3) ): Cartesian coordinates of the atoms.

    - ref_atoms (list, size: (n_atoms, 3) ): Reference atoms for internal coordinates generation. The distance will be computed with respect the first atom of the triplets,
                                             the angle with respect to the first and the second and the torsion with respect to the three. 

    - placing_order (list, size: (n_atoms) ): placing order of the atoms.

    Returns: 

    - z_matrix (torch.tensor, size: (n_atoms-1, 3) ): Z-Matrix of the configuration of the molecule.  

    '''
    if placing_order is None:
        placing_order = list(range(len(ref_atoms)))

    torch.pi = torch.tensor(np.pi)
    i3 = [triplet[0] for triplet in ref_atoms]
    i2 = [triplet[1] for triplet in ref_atoms]
    i1 = [triplet[2] for triplet in ref_atoms]

    # Mind the order!
    x4 = X[placing_order]
    x3 = X[i3[1:]]
    x2 = X[i2[2:]]
    x1 = X[i1[3:]]

    distances = compute_distance(x4[1:], x3)
    angles = compute_angle(x4[2:], x3[1:], x2)
    torsions = compute_torsion(x1, x2[1:], x3[2:], x4[3:])

    n_atoms = X.shape[0]
    z_matrix = torch.zeros([n_atoms -1 , 3], dtype = torch.float32)

    z_matrix[:, 0] = distances 
    z_matrix[1:, 1] = angles
    z_matrix[2:, 2] = torsions

    return z_matrix

def construct_z_matrix_batch(X_batch, ref_atoms, placing_order=None):
    '''
    Given a torch.tensor array, whose rows are the cartesian coordinates of a molecule, generates a Z-matrix computing internal coordinates.

    Arguments:

    - X (torch.tensor, size: (n_conformations, n_atoms, 3) ): Cartesian coordinates of the atoms.

    - ref_atoms (list, size: (n_atoms, 3) ): Reference atoms for internal coordinates generation. The distance will be computed with respect the first atom of the triplets,
                                             the angle with respect to the first and the second and the torsion with respect to the three. 

    - placing_order (list, size: (n_atoms) ): placing order of the atoms.

    Returns: 

    - z_matrix (torch.tensor, size: (n_atoms-1, 3) ): Z-Matrix of the configuration of the molecule.  

    '''
    if placing_order is None:
        placing_order = list(range(len(ref_atoms)))

    torch.pi = torch.tensor(np.pi)
    n_atoms = X_batch.shape[1]

    i3 = [triplet[0] for triplet in ref_atoms]
    i2 = [triplet[1] for triplet in ref_atoms]
    i1 = [triplet[2] for triplet in ref_atoms]

    # Mind the order!

    x4 = X_batch[:,placing_order,:]
    x3 = X_batch[:, i3[1:], :]
    x2 = X_batch[:, i2[2:], :]
    x1 = X_batch[:, i1[3:], :]

    distances = compute_distance(x4[:,1:,:], x3)
    angles = compute_angle(x4[:,2:,:], x3[:,1:,:], x2)
    torsions = compute_torsion(x1, x2[:,1:, :], x3[:,2:,:], x4[:,3:,:])

    
    z_matrix = torch.zeros([X_batch.size()[0], n_atoms -1 , 3], dtype = torch.float32)

    z_matrix[:, :, 0] = distances 
    z_matrix[:, 1:, 1] = angles
    z_matrix[:, 2:, 2] = torsions

    return z_matrix

def deconstruct_z_matrix(z_matrix, ref_atoms, jacobian=False, loss=False):
    '''
    Generates cartesian coordinates given a z-matrix and the reference atoms. Requires the z_matrix to be correctly sorted.

    Arguments:
        - z_matrix (torch.tensor, size: (n_atoms-1, 3) )
        - ref_atoms (list, size: (n_atoms, 3) ): List of the reference atoms.

    Returns:
        - cartesian (torch.tensor, size: (n_atoms, 3) ): Coordinates using a nerf ref system.
        - constraints loss (torch.tensor, size (n_atoms))
    '''
    torch.pi = torch.tensor(np.pi)
    z_matrix_local = z_matrix.clone() 
    n_atoms = len(ref_atoms)
    if loss:
        def _dist_loss(dists):
            loss = torch.sum(torch.where(dists<0, dists, torch.zeros_like(dists))**2, axis=-1) 
            return loss

        def _polar_loss(angles):
            negative_loss = torch.sum(torch.where( angles < 0, - angles, torch.zeros_like(angles)) ** 2, axis=-1)
            positive_loss = torch.sum(torch.where( angles > torch.pi, angles - torch.pi, torch.zeros_like(angles)) ** 2, axis=-1)
            return  negative_loss + positive_loss

        def _torsion_loss(angles):
            negative_loss = torch.sum( torch.where(angles < - torch.pi, angles + torch.pi, torch.zeros_like(angles) )**2, axis = -1 )
            positive_loss = torch.sum( torch.where(angles > torch.pi, angles - torch.pi, torch.zeros_like(angles) )**2 )
            return  negative_loss + positive_loss 

        internal_constraints_loss = _dist_loss( z_matrix_local[:,0].clone() ) + _polar_loss( z_matrix_local[:,1].clone() ) + _torsion_loss( z_matrix_local[:,2].clone() )

    # Protection
    z_matrix_local[:,0] = torch.clamp(z_matrix_local[:,0].clone(), min =0)
    z_matrix_local[:,1] = torch.clamp(z_matrix_local[:,1].clone(), min =0, max = np.pi )
    #z_matrix_local[:,2] = torch.clamp(z_matrix_local[:,2].clone(), min = - np.pi, max = np.pi )

    if jacobian:
        logdetJ = 0

    cartesian = torch.zeros( (n_atoms, 3), dtype = torch.float32, device = z_matrix_local.device)
    cartesian[1] = z_matrix_local[0].clone()

    if ref_atoms[2][0]:
        angle = torch.pi - z_matrix[1,1]
        a_sign = -1
    else:
        angle = z_matrix[1,1]
        a_sign = 1
    cartesian[2,0] = cartesian[ ref_atoms[2][0] , 0] + z_matrix[1,0] * torch.cos(angle)
    cartesian[2,1] = z_matrix[1,0] * torch.sin(angle)

    if jacobian:
        J_2 = torch.zeros((2,2), dtype = torch.float32, device = z_matrix_local.device)
        J_2[0,0] = torch.cos(angle)
        J_2[0,1] =  -a_sign*z_matrix[1,0] * torch.sin(angle)
        J_2[1,0] = torch.sin(angle)
        J_2[1,1] =  a_sign*z_matrix[1,0] * torch.cos(angle)

        logdetJ += torch.log(torch.abs(torch.det(J_2)))

    for i_atom in range(3, n_atoms):
        x_ref = cartesian[ list(ref_atoms[i_atom]) ]
        if jacobian:
            cartesian[i_atom], detJ_i = ic_to_xyz(x_ref[2].unsqueeze(dim = 0), x_ref[1].unsqueeze(dim = 0), x_ref[0].unsqueeze(dim = 0),
                                      z_matrix_local[i_atom-1, 0].unsqueeze(dim = 0), z_matrix_local[i_atom-1, 1].unsqueeze(dim = 0), z_matrix_local[i_atom-1, 2].unsqueeze(dim = 0), jacobian=jacobian)
            logdetJ += torch.log(torch.abs(detJ_i[0]))
        else:
            cartesian[i_atom] = ic_to_xyz(x_ref[2].unsqueeze(dim = 0), x_ref[1].unsqueeze(dim = 0), x_ref[0].unsqueeze(dim = 0),
                                      z_matrix_local[i_atom-1, 0].unsqueeze(dim = 0), z_matrix_local[i_atom-1, 1].unsqueeze(dim = 0), z_matrix_local[i_atom-1, 2].unsqueeze(dim = 0), jacobian=jacobian)
    if loss and jacobian:
      return cartesian, logdetJ, internal_constraints_loss 
    elif jacobian:
      return cartesian, logdetJ
    elif loss:
        return cartesian, loss
    else:
        return cartesian

def deconstruct_z_matrix_batch(z_matrices, ref_atoms, jacobian=False, angle_tolerance=1e-5):
    '''
    Generates cartesian coordinates given a z-matrix and the reference atoms. Requires the z_matrix to be correctly sorted.

    Arguments:
        - z_matrices (torch.tensor, size: (n_batch, n_atoms-1, 3) )
        - ref_atoms (list, size: (n_atoms, 3) ): List of the reference atoms.

    Returns:
        - cartesian (torch.tensor, size: (n_batch, n_atoms, 3) ): Coordinates using a nerf ref system.
        - constraints loss (torch.tensor, size (n_atoms))
    '''
    torch.pi = torch.tensor(np.pi)
    z_matrices_local = z_matrices.clone() 
    n_atoms = len(ref_atoms)
    n_conf = z_matrices_local.size()[0]

    if jacobian:
        logdetJ = torch.zeros(n_conf, device = z_matrices_local.device)

    cartesian = torch.zeros( (n_conf, n_atoms, 3), dtype = torch.float32, device = z_matrices_local.device)
    cartesian[:,1,0] = z_matrices_local[:,0,0]

    if ref_atoms[2][0]:
        angle = torch.pi - z_matrices_local[:,1,1]
        a_sign = -1
    else:
        angle = z_matrices_local[:,1,1]
        a_sign = 1
    cartesian[:,2,0] = cartesian[:, ref_atoms[2][0] , 0] + z_matrices_local[:,1,0] * torch.cos(angle)
    cartesian[:,2,1] = z_matrices_local[:,1,0] * torch.sin(angle)

    if jacobian:
        J_2 = torch.zeros((n_conf,2,2), dtype = torch.float32, device = z_matrices_local.device)
        J_2[:,0,0] = torch.cos(angle)
        J_2[:,0,1] =  -a_sign*z_matrices_local[:,1,0] * torch.sin(angle)
        J_2[:,1,0] = torch.sin(angle)
        J_2[:,1,1] =  a_sign*z_matrices_local[:,1,0] * torch.cos(angle)

        logdetJ += torch.log(torch.abs(torch.det(J_2)))

    for i_atom in range(3, n_atoms):
        x_ref = cartesian[:, list(ref_atoms[i_atom]) ]
        if jacobian:
            cartesian[:,i_atom], detJ_i = ic_to_xyz(x_ref[:,2], x_ref[:,1], x_ref[:,0],
                                      z_matrices_local[:,i_atom-1, 0], z_matrices_local[:,i_atom-1, 1], z_matrices_local[:,i_atom-1, 2], jacobian=jacobian)
            logdetJ += torch.log(torch.abs(detJ_i))
        else:
            cartesian[:,i_atom]= ic_to_xyz(x_ref[:,2], x_ref[:,1], x_ref[:,0],
                                      z_matrices_local[:,i_atom-1, 0], z_matrices_local[:,i_atom-1, 1], z_matrices_local[:,i_atom-1, 2], jacobian=jacobian)
    
    if jacobian:
      return cartesian, logdetJ
    else:
        return cartesian


def compute_jacobian_batch(z_matrices, ref_atoms):
    '''
    Generates cartesian coordinates given a z-matrix and the reference atoms. Requires the z_matrix to be correctly sorted.

    Arguments:
        - z_matrices (torch.tensor, size: (n_batch, n_atoms-1, 3) )
        - ref_atoms (list, size: (n_atoms, 3) ): List of the reference atoms.

    Returns:
        - cartesian (torch.tensor, size: (n_batch, n_atoms, 3) ): Coordinates using a nerf ref system.
        - constraints loss (torch.tensor, size (n_atoms))
    '''
    torch.pi = torch.tensor(np.pi)
    n_atoms = len(ref_atoms)
    n_conf = z_matrices.size()[0]
    device = z_matrices.device

    logdetJ = torch.zeros(n_conf, device = device)

    if ref_atoms[2][0]:
        angle = torch.pi - z_matrices[:,1,1]
        a_sign = -1
    else:
        angle = z_matrices[:,1,1]
        a_sign = 1

    J_2 = torch.zeros((n_conf,2,2), dtype = torch.float32, device = device)
    J_2[:,0,0] = torch.cos(angle)
    J_2[:,0,1] =  -a_sign*z_matrices[:,1,0] * torch.sin(angle)
    J_2[:,1,0] = torch.sin(angle)
    J_2[:,1,1] =  a_sign*z_matrices[:,1,0] * torch.cos(angle)

    logdetJ += torch.log(torch.abs(torch.det(J_2)))

    for i_atom in range(3, n_atoms):

        d34 = z_matrices[:,i_atom-1, 0]
        a234 = z_matrices[:,i_atom-1, 1]
        t1234 = z_matrices[:,i_atom-1, 2]


        J_det = torch.zeros((d34.size()[0],3,3), dtype = torch.float32).to(device)
        J_det[:, 0, 0] = torch.cos(torch.pi-a234)
        J_det[:, 0, 1] = z_matrices[:,i_atom-1, 0] * torch.sin(torch.pi-a234) #note: - sign because we use pi-a234

        J_det[:, 1, 0] = torch.sin(torch.pi-a234) * torch.cos(t1234)
        J_det[:, 1, 1] = -d34 * torch.cos(torch.pi-a234) * torch.cos(t1234)#note: - sign because we use pi-a234
        J_det[:, 1, 2] = -d34 * torch.sin(torch.pi-a234) * torch.sin(t1234) 

        J_det[:, 2, 0] = torch.sin(torch.pi-a234) * torch.sin(t1234)
        J_det[:, 2, 1] = -d34 * torch.cos(torch.pi-a234) * torch.sin(t1234)#note: - sign because we use pi-a234
        J_det[:, 2, 2] = d34 * torch.sin(torch.pi-a234) * torch.cos(t1234)


        logdetJ += torch.log(torch.abs(torch.det(J_det)))
   
    return  logdetJ


def correct_conf_indexes(conformations):
    '''
        conformations (torch.tensor): Z-matrixes of the conformations.
    '''
    torch.pi = torch.tensor(np.pi)
    indexes = []
    for i_conf, conformation in enumerate(conformations):
        d_i = conformation[:,0] > 0
        d_a =  torch.logical_and(conformation[:,1] >= 0, conformation[:,1] <= torch.pi)
        d_t =  torch.logical_and(conformation[:,2] > - torch.pi, conformation[:,2] <= torch.pi)
        if torch.logical_not(torch.any( torch.logical_not(torch.logical_and(torch.logical_and(d_i, d_a), d_t)) )): #Simplify this
            indexes.append(i_conf)
    return indexes

