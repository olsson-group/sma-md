import torch
import numpy as np

torch.pi = torch.tensor(np.pi)

def torchdot(x1, x2, dim = -1):
    '''
    Computes dot product of the rows (dim = 1) or columns (dim = 0) vectors of x1 and x2.

    Arguments_

    - x1, x2 (torchtensors)

    Outputs:
    Row (1xN) or column (Nx1) tensor  with values of the dot product of the vectors
    '''
    return torch.sum(x1*x2, dim = dim)


def compute_distance(x1, x2):
    '''
    Computes the distance between two points.

    Arguments:

     -x1, x2 (torch tensors, size: (N,3) ): Cartesian coordinates (row) of the 2 atoms. If several rows, each row is the positions of the 1st (x1) and 2nd (x2) point.

     Outputs:

     -distances (torch.tensor, size: (N) ): Row tensor of distances between tuples of atoms.
    '''
    return torch.norm(x2-x1, dim = -1)


def compute_angle(x1, x2, x3):
    '''
    Computes the angle genearated by three atoms in radians.

    Arguments:

     -x1,x2,x3 (torch tensor, size: (N,3) ): Cartesian coordinates of the 3 atoms. If several rows, respectevily 1st, 2nd, 3rd positions of the triplet at each row.

     Outputs:

     -angles (torch.tensor, size: (N)): Row tensor of angles defined by triplets of atoms.
    '''
    x21 = x1-x2
    x23 = x3-x2
    angles = torch.acos( torchdot(x21, x23, dim = -1) / (    torch.norm(x21, dim = -1) * torch.norm(x23, dim = -1)) )
    return angles

  
def compute_torsion(x1, x2, x3, x4):
    '''
    Computes the torsion, given 4 (in order) atoms, in radians using the atan2 function. The range of the torsions is (-pi,pi].

        Arguments:

        -x1,x2,x3 and x4 (torch tensor, size: (n_atoms, 3) ): Cartesian coordinates of the 4 atoms. If several rows, respectevily 1st, 2nd, 3rd and 4rd positions of the quartet at each row.

        Outputs:

     -torsions (torch.tensor, size: (n_atoms) ): Row tensor of torsions defined by quartets of atoms.
    '''
    x12 = x2-x1
    x23 = x3-x2
    x34 = x4-x3
    
    cross_23_34 = torch.cross(x23, x34, dim = -1)
    y = torch.norm(x23, dim =-1) * torchdot(x12, cross_23_34, dim =-1)
    x =  torchdot( torch.cross(x12, x23, dim =-1), cross_23_34, dim=-1)

    torsions = torch.atan2(y, x)

    return torsions

def index_compute_torsion(X, i, j, k, l):
    '''
    Computes the torsion, given 4 (in order) atoms, in radians using the atan2 function.

    - Arguments:

        - X (torch tensor, size: (n_atoms, 3) ): Cartesian coordinates of the molecule.
        - i , j, k and l (list (NOT TUPPLE)): indexes of the, respectively, first, second, third and fourth atoms to compute the torsions.

    - Outputs:

        -torsions (torch.tensor, size: (n_atoms, 3) ): Row tensor of torsions defined by quartets of atoms.
    '''
    x1 = X[i]
    x2 = X[j]
    x3 = X[k]
    x4 = X[l]

    x12 = x2-x1
    x23 = x3-x2
    x34 = x4-x3

    y = torch.norm(x23, dim = 1) * torchdot(x12, torch.cross(x23, x34, dim= 1))
    x =  torchdot( torch.cross(x12, x23, dim =1), torch.cross(x23, x34, dim = 1)  )

    torsions = torch.atan2(y, x)

    return torsions



def ic_to_xyz(p1, p2, p3, d34, a234, t1234, jacobian = False):
    '''
    Computes the cartesian coordinates of atom 4 given the cartesian coordinates of the other three atoms and its corresponding internal coordinates. p3, 2 and 1 are respectively the distance, angle and torsion references.
    - Arguments:
        - p1, p2 and p3 (torch.tensors, size: (n_atoms,3) ): Cartesian coordinates of the three reference atoms.
        - d34, a234 and t1234 ((torch.tensors, size: (n_atoms) )): distances, angles and torsions of the atoms to place.
        - jacobian (bool): If True returns the determinant of the jacobian of the transformation as well.
    - Returns:
        - positions (torch.tensors, size: (n_atoms,3) ): Position of the atoms in the original cartesian system.
    '''

    device = p1.device
    D_prima = torch.zeros((d34.size()[0],3,1), dtype = torch.float32).to(device)
    D_prima[:, 0, 0] = d34 * torch.cos(torch.pi-a234)
    D_prima[:, 1, 0] = d34 * torch.sin(torch.pi-a234) * torch.cos(t1234)  
    D_prima[:, 2, 0] = d34 * torch.sin(torch.pi-a234) * torch.sin(t1234)

    if jacobian:
        #note: J_det is not the actual Jacobian, is an easy-to-copmute matrix that has the same determinant.
        J_det = torch.zeros((d34.size()[0],3,3), dtype = torch.float32).to(device)
        J_det[:, 0, 0] = torch.cos(torch.pi-a234)
        J_det[:, 0, 1] = d34 * torch.sin(torch.pi-a234) #note: - sign because we use pi-a234

        J_det[:, 1, 0] = torch.sin(torch.pi-a234) * torch.cos(t1234)
        J_det[:, 1, 1] = -d34 * torch.cos(torch.pi-a234) * torch.cos(t1234)#note: - sign because we use pi-a234
        J_det[:, 1, 2] = -d34 * torch.sin(torch.pi-a234) * torch.sin(t1234) 

        J_det[:, 2, 0] = torch.sin(torch.pi-a234) * torch.sin(t1234)
        J_det[:, 2, 1] = -d34 * torch.cos(torch.pi-a234) * torch.sin(t1234)#note: - sign because we use pi-a234
        J_det[:, 2, 2] = d34 * torch.sin(torch.pi-a234) * torch.cos(t1234)

        #J = M@J_det #is the Jacobian

    p23 = p3 - p2
    x23 = p23/torch.norm(p23, dim = 1, keepdim = True)
    x12 = p2 - p1
    #x12 /= torch.norm(x12, dim = 1, keepdim = True) 

    cross_12_23 = torch.cross(x12, x23, dim = 1)
    n = cross_12_23/torch.norm(cross_12_23, dim = 1, keepdim = True)

    M = torch.zeros((d34.size()[0],3,3), dtype = torch.float32).to(device)
    M[:,:,0] = x23
    M[:,:,1] = torch.cross(n, x23, dim = 1)
    M[:,:,2] = n

    positions = p3 + torch.bmm(M, D_prima).squeeze(dim = -1)

    if jacobian:
        return positions, torch.det(J_det)
    else:
        return positions

def jacobian_atom(d34, a234, t1234, device):#Cam we batch this?
    """

    Args:
        d34 (_type_): _description_
        a234 (_type_): _description_
        t1234 (_type_): _description_
    """
    J_det = torch.zeros((d34.size()[0],3,3), dtype = torch.float32).to(device)
    J_det[:, 0, 0] = torch.cos(torch.pi-a234)
    J_det[:, 0, 1] = d34 * torch.sin(torch.pi-a234) #note: - sign because we use pi-a234

    J_det[:, 1, 0] = torch.sin(torch.pi-a234) * torch.cos(t1234)
    J_det[:, 1, 1] = -d34 * torch.cos(torch.pi-a234) * torch.cos(t1234)#note: - sign because we use pi-a234
    J_det[:, 1, 2] = -d34 * torch.sin(torch.pi-a234) * torch.sin(t1234) 

    J_det[:, 2, 0] = torch.sin(torch.pi-a234) * torch.sin(t1234)
    J_det[:, 2, 1] = -d34 * torch.cos(torch.pi-a234) * torch.sin(t1234)#note: - sign because we use pi-a234
    J_det[:, 2, 2] = d34 * torch.sin(torch.pi-a234) * torch.cos(t1234)

    return torch.det(J_det)

def ic_to_xyz_test(p1, p2, p3, d34, a234, t1234):
    '''
    Computes the cartesian coordinates of atom 4 given the cartesian coordinates of the other three atoms and its corresponding internal coordinates. p3, 2 and 1 are respectively the distance, angle and torsion references.
    - Arguments:
        - p1, p2 and p3 (torch.tensors, size: (n_atoms,3) ): Cartesian coordinates of the three reference atoms.
        - d34, a234 and t1234 ((torch.tensors, size: (n_atoms,3) )): distances, angles and torsions of the atoms to place.
    - Returns:
        - positions (torch.tensors, size: (n_atoms,3) ): Position of the atoms in the original cartesian system.
    '''

    device = p1.device
    x4_int = torch.zeros((3), dtype = torch.float32).to(device)
    x4_int[0] = d34 * torch.cos(a234)
    x4_int[1] = d34 * torch.sin(a234) * torch.cos(t1234)  
    x4_int[2] = d34 * torch.sin(a234) * torch.sin(t1234)

    x23 = p3 - p2
    x23 /= torch.norm(x23)
    x12 = p2 - p1
    #x12 /= torch.norm(x12) 

    n = torch.cross(x12, x23)
    n /= torch.norm(n)

    R = torch.zeros((3,3), dtype = torch.float32).to(device)
    R[:,0] = x23
    R[:,1] = torch.cross(n, x23)
    R[:,2] = n

    positions = p3 + torch.matmul(R, x4_int)

    return positions