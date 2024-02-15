import torch
import numpy as np
import copy
from rdkit import Chem

from utils.local_geometry import compute_terminals

def dfs(neighbors, node=0, visited=None):
    '''

    '''
    if visited == None: visited = []
    if node not in visited:
        visited.append(node)
        for neighbor in neighbors[node]: #note neighbors should be sorted in the ranking order
            dfs(neighbors, neighbor, visited)

    return visited

def bfs(neighbors, node=0, visited=None):
    '''
    
    '''
    if visited == None: visited = []
    queue = []
    #visited.append(node) #for ease in this implementation. This line would be needed in the original algorithm.
    queue.append(node)

    while queue:
        center = queue.pop(0)
        for neighbor in neighbors[center]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)

    return visited

def bfs_parents(neighbors, node=0, visited=None, parents = None):
    '''
    
    '''
    if visited == None: visited = []
    if parents == None: parents = [None]
    queue = []
    #visited.append(node) #for ease in this implementation. This line would be needed in the original algorithm.
    queue.append(node)

    while queue:
        center = queue.pop(0)
        for neighbor in neighbors[center]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                parents.append(center)

    return visited, parents

def find_first_references(neighbors):
    '''
        Look for a path of, preferably, 4 nodes in a graph. If it doesn't exist, find the longest path.
        Input:
            - neighbors: list whose ith entry are the neighbours of node i.
        Output:
            - path
            - path_length: if 0, the longest path is shorter than 3.

    '''
    #could be optimized to not repeat paths using sets.
    path_length = 0
    length_3_paths = []
    for node_1 in range(len(neighbors)):
        for node_2 in neighbors[node_1]:
            for node_3 in neighbors[node_2]:
                if node_3 != node_1:
                    length_3_paths.append([node_1, node_2, node_3])
                    path_length = 3
                    for node_4 in neighbors[node_3]:
                        if node_4 != node_2 and node_4 != node_1:
                            path_length = 4
                            return [node_1, node_2, node_3, node_4], path_length
    for path in length_3_paths:
        if len(neighbors[path[1]])==2:
            return path, 3
    return [], 0

def compute_ref_atoms_sorted_molecule(molecule, n_non_exceptional_atoms, terminal_groups):
    '''
    Sorted here means that terminal atoms are last.

    '''
    n_nodes = molecule.GetNumAtoms()
    A = Chem.GetAdjacencyMatrix(molecule)
    n_atoms = len(A)
    neighbors = []
    for i_node in range(n_nodes):
        neighbors.append([int(i) for i in np.nonzero(A[i_node, :])[0]])

    ref_atoms = [ [] for _ in range(n_nodes) ]
    ref_atoms[0] = [None, None, None]
    ref_atoms[1] = [0, None, None]
    if n_nodes > 2:
        if 0 in [int(i) for i in np.nonzero(A[2, :])[0]]:
            ref_atoms[2] = [0, 1, None]
        else:
            ref_atoms[2] = [1, 0, None]

    if n_nodes > 3:
        for i_node in range(3,n_non_exceptional_atoms):
            success = False
            r3_candidates = [int(i) for i in np.nonzero(A[i_node, :])[0] if i<i_node]
            r3_candidates.sort()
            if not r3_candidates:
                r3_candidates = []
            for r3_candidate in r3_candidates:
                r2_candidates = [int(i) for i in np.nonzero(A[r3_candidate, :])[0] if i<i_node]
                r2_candidates.sort()
                if not r2_candidates:
                    r2_candidates = []
                for r2_candidate in r2_candidates:
                    r1_candidates = [int(i) for i in np.nonzero(A[r2_candidate, :])[0] if i<i_node and i != r3_candidate]
                    if r1_candidates:
                        r3 = r3_candidate
                        r2 = r2_candidate
                        r1 = min(r1_candidates)
                        success = True
                        break   
                if success:
                    break
            if success:
                ref_atoms[i_node] = [r3, r2, r1]
            else:
                #print("The graph transversing algorithm found an exception computing reference atoms.")
                r3 = min([int(i) for i in np.nonzero(A[i_node, :])[0] if i < i_node])
                r2 = min([int(i) for i in np.nonzero(A[r3, :])[0] if i < i_node])
                r1 = min([int(i) for i in range(i_node) if i!=r3 and i!=r2])
                ref_atoms[i_node] = [r3, r2, r1]
        for group in terminal_groups:
            if len(group) > 1:
                for i_node in group[1:]:
                    r3 = np.nonzero(A[i_node])[0][0]
                    r1 = group[0]
                    r2_candidates = [int(i) for i in np.nonzero(A[r3])[0] if i not in group]
                    if len(r2_candidates) == 1:
                        r2 = r2_candidates[0]
                    else:
                        r2 = list(r2_candidates)

                    ref_atoms[i_node] = [r3, r2, r1]

    return ref_atoms

def compute_atom_order_and_references(molecule):
    '''
        Computes a convient atom placing order (based on a BF algorithm over the non-terminal atoms) 
        and the corresponding reference atoms for transBG.

        Input:
            - molecule (rdkit molecule)
        Output:
            - node_order_original_indexes (list): atom sorting wiht respect to the original sorting.
            - ref_atoms(list of lists): reference atoms with respect to final sorting.

        WARNING: ref_atoms is given with respect of the final sorting of the atoms.
    '''
    
    n_nodes = molecule.GetNumAtoms()
    A = Chem.GetAdjacencyMatrix(molecule)
    terminal_atoms, terminal_groups, terminal_groups_centers, _ = compute_terminals(molecule)
    #make sure terminal atoms are last
    terminals_original = []
    non_terminals_original = []
    for i_atom in range(len(A)):
        if np.sum(A[i_atom]) == 1:
            terminals_original.append(i_atom)
        else:
            non_terminals_original.append(i_atom) #copmute this more efficiently using terminals_original

    #transverse the graph after finding a proper starting point
    A = Chem.GetAdjacencyMatrix(molecule)
    A_non_terminal = A[non_terminals_original][:, non_terminals_original]
    
    neighbors = []
    for i_node in range(len(A_non_terminal)):
        neighbors.append([int(i) for i in np.nonzero(A_non_terminal[i_node, :])[0]])

    node_order = [0]
    bfs(neighbors, node=0, visited=node_order) #note: node_order is mutable and changes during the loop
    node_order_copy = node_order.copy()
    node_order = [non_terminals_original[node_order_copy[i_atom]] for i_atom in range(len(node_order_copy))]
    #resort terminals
    terminal_atoms = []
    for group in terminal_groups:
        terminal_atoms.append(group[0])
    n_non_exceptional_atoms = len(node_order.copy() + terminal_atoms.copy())
    for group in terminal_groups:
         if len(group) > 1:
                for i_node in group[1:]:
                    terminal_atoms.append(i_node)
    node_order += terminal_atoms #bfs over non-terminals + terminals

    #sort and re-compute quantities in the new order
    molecule = Chem.RenumberAtoms(molecule, node_order)
    
    #these elemts can be inverted for speed instead of re-computing them
    terminal_atoms, terminal_groups, terminal_groups_centers, _ = compute_terminals(molecule)

    ref_atoms = compute_ref_atoms_sorted_molecule(molecule, n_non_exceptional_atoms, terminal_groups)
    
    return node_order, ref_atoms

def compute_atom_order_and_references_groups(molecule):
    '''
        Computes a convient atom placing order (based on a BF algorithm over the non-terminal atoms) 
        and the corresponding reference atoms for transBG.

        Input:
            - molecule (rdkit molecule)
        Output:
            - node_order_original_indexes (list): atom sorting wiht respect to the original sorting.
            - ref_atoms(list of lists): reference atoms with respect to final sorting.

        WARNING: ref_atoms is given with respect of the final sorting of the atoms.
    '''
    n_atoms = molecule.GetNumAtoms()
    A = Chem.GetAdjacencyMatrix(molecule)
    neighbors = []
    for i_node in range(n_atoms):
        neighbors.append([int(i) for i in np.nonzero(A[i_node, :])[0]])
    
    if n_atoms == 2:
        groups = []
        atom_order = [0,1]
        ref_atoms = [[None, None, None], [0, None, None]]
        return groups, atom_order, ref_atoms


    #get non-terminals and connectivity
    non_terminals = [ i_atom for i_atom in range(n_atoms) if np.sum(A[i_atom]) > 1 ]
    A_non_terminal = A[non_terminals][:, non_terminals]
    nt_neighbors = []
    for i_node in range(len(non_terminals)):
        nt_neighbors.append([non_terminals[int(i)] for i in np.nonzero(A_non_terminal[i_node, :])[0]])


    #choose a starting point that is a semi-terminal node. It is not needed, but the assembly part is more intuitive this way.
    if len(non_terminals) > 0:
        starting_point = 0
        for non_terminal in non_terminals:
            atom_neighbors = neighbors[non_terminal]
            aux = [ np.sum(A[neighbor]) == 1 for neighbor in atom_neighbors ]
            if np.sum(aux) == len(aux)-1 or np.sum(aux) == len(aux):
                starting_point = non_terminals.index(non_terminal)
                break
    else:
        starting_point = 0
    

    #bfs over non-terminals
    if len(non_terminals) > 1:
        nt_neighbors_nt_indexes = []
        for atom_neighbors in nt_neighbors:
            atom_neighbors_nt_indexes = []
            for neighbor in atom_neighbors:
                atom_neighbors_nt_indexes.append(non_terminals.index(neighbor))
            nt_neighbors_nt_indexes.append(atom_neighbors_nt_indexes)
        nt_order = [starting_point]
        visited_nt_indexes, parents_nt_indexes = bfs_parents(nt_neighbors_nt_indexes, node=starting_point, visited=nt_order)#check this, compute and save precessor atom of each group
        nt_order = list(np.array(non_terminals)[visited_nt_indexes])
        parents = [None]+list(np.array(non_terminals)[parents_nt_indexes[1:]])
    else:
        nt_order = [non_terminals[0]]
        parents = [None]

    #make groups (sorting their elements) and compute ref_atoms

    groups = []
    atom_order = [int(nt_order[0])]
    ref_atoms = []

    #first group
    atom_neighbors = neighbors[nt_order[0]]
    neg_n_neighbors = [ -np.sum(A[i_neighbor]) for i_neighbor in atom_neighbors ]
    sorted_neighbors = [ int(i_neighbor) for _, i_neighbor in sorted(zip(neg_n_neighbors, atom_neighbors)) ]
    atom_order += sorted_neighbors
    groups.append([nt_order[0]] + sorted_neighbors)
    ref_atoms = [ [None, None, None], [nt_order[0], None, None], [ nt_order[0], sorted_neighbors[0], None ] ]
    for _ in sorted_neighbors[2:]:
        ref_atoms.append([ nt_order[0], sorted_neighbors[0], sorted_neighbors[1] ])

    for non_terminal, parent in zip(nt_order[1:], parents[1:]):
        atom_neighbors = neighbors[non_terminal].copy()
        atom_neighbors.remove(parent)
        neg_n_neighbors = [ -np.sum(A[i_neighbor]) for i_neighbor in atom_neighbors ]
        sorted_neighbors = [ i_neighbor for _, i_neighbor in sorted(zip(neg_n_neighbors, atom_neighbors)) ]

        group = [non_terminal] + [parent] + sorted_neighbors
        groups.append(group)
        for i_neighbor, atom in enumerate(sorted_neighbors):
            if atom not in(atom_order):#if we protect for cycles this can be not needed.
                atom_order.append(int(atom))
                if i_neighbor == 0:
                    neigh_of_parent = neighbors[parent].copy()
                    neigh_of_parent.remove(non_terminal)
                    thrid_ref = neigh_of_parent[0]
                    ref_atoms.append( [non_terminal, parent, thrid_ref] )
                else:    
                    ref_atoms.append( [non_terminal, parent, sorted_neighbors[0]] )
    #get ref atoms and groups in the new ordering
    inverse = [ atom_order.index(i_atom) for i_atom in range(len(atom_order)) ]

    ref_atoms_old_order = copy.deepcopy(ref_atoms)
    ref_atoms = [ [None, None, None], [0, None, None]]
    ref_atoms.append( [ inverse[ref_atoms_old_order[2][0]], inverse[ref_atoms_old_order[2][1]], None ] )
    for i_atom in range(3, n_atoms):
        ref_atoms.append( [ inverse[ref_atoms_old_order[i_atom][0]], inverse[ref_atoms_old_order[i_atom][1]], inverse[ref_atoms_old_order[i_atom][2]] ] )
        
    groups_old_order = copy.deepcopy(groups)
    groups = []
    for group in groups_old_order:
        new_group = []
        for atom in group:
            new_group.append(inverse[atom])
        groups.append(new_group.copy())

    return atom_order, groups, ref_atoms

def general_compute_ref_atoms(molecule, mode = 'bf'):
    '''
        Computes reference atoms for computing internal coordinates using a breadth(bf)- or depth(df)-first algorithm. Assumes the molecule is ordered according to an atom ranking.
        Returns: ref_atoms (list): reference atoms.
    '''
    assert mode == 'df', "DFS is not currently implemented"

    assert mode == "bf", "BFS is not currently implemented."

    return []


#used for AI for molecules workshop

def breadth_first_sorting(molecule, node_ranking : list, node_init : int=0):
    '''
        Starting from the specified `node_init` in the graph, uses a breadth-first 
        search (BFS) algorithm to find all adjacent nodes, returning an ordered list 
        of these nodes. Prioritizes the nodes based on the input `node_ranking`. 
        Args:
        ----
            node_ranking (list) : Contains the ranking of all the nodes in the 
                                  graph (e.g. the canonical RDKit node ranking,
                                  or a random ranking).
            node_init (int) : Index of node to start the BFS from. Default 0.
        Returns:
        -------
            nodes_visited (list) : BFS ordering for nodes in the molecular graph.
            ref_atoms (list): Possible 3 reference atoms for computing internal coordinates.
    '''

    nodes_visited = [node_init]
    last_nodes_visited = [node_init]
    ref_atoms = [ [] for _ in range(molecule.GetNumAtoms()) ]
    
    A = torch.tensor(Chem.GetAdjacencyMatrix(molecule), dtype = torch.int32)

    ref_atoms[node_init] = [-1, -1, -1]


    # loop until all nodes have been visited
    while len(nodes_visited) < molecule.GetNumAtoms():
        neighboring_nodes = []
        
        for node in last_nodes_visited:
            neighbor_nodes = [int(i) for i in torch.nonzero(A[node, :])]
            new_neighbor_nodes = list(
                set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
            )
            node_importance = [node_ranking[neighbor_node] for
                                neighbor_node in new_neighbor_nodes]

            # check all neighboring nodes and sort in order of importance
            while sum(node_importance) != -len(node_importance):
                next_node = node_importance.index(max(node_importance))
                neighboring_nodes.append(new_neighbor_nodes[next_node])
                node_importance[next_node] = -1

            #Finally protect from reaching to the same atom at the same time
            neighboring_nodes = list(set(neighboring_nodes))
        #Generate reference atoms:
        for new_neighbor_node in neighboring_nodes:
            ref = []
            new_ref = new_neighbor_node #Juts for coding convenience

            #Try to construct a chain of three important connected atoms
            while len(ref) < 3:
                neigh_neigh_nodes = [int(i) for i in torch.nonzero(A[new_ref, :])]
                possible_new_ref = list( (set(neigh_neigh_nodes) & set(nodes_visited)) - set(ref) )
                if possible_new_ref == []:
                    break # This is improvable, add a counter with max the number of options to start the chain and try other neighbors. Otherwise just use the very last and legal ones.
                max_rank = max( [node_ranking[i] for i in possible_new_ref] )
                new_ref = node_ranking.index(max_rank)
                ref.append( new_ref )
            if len(ref) == 3:   # If we managed to build a proper chain, use it as reference
                ref_atoms[new_neighbor_node] = ref.copy()

            else:
                if len(nodes_visited) > 2: # If we have enough atoms, just take the last placed and not used.
                    non_used_atoms = list( set(set(nodes_visited) ) - set(ref) )
                    while len(ref) < 3:
                        ref.append(non_used_atoms[len(ref)-3])                         
                    ref_atoms[new_neighbor_node] = ref.copy()
                else: #If we do not have enough visited atoms, just add dummy tokens -1
                    ref = nodes_visited.copy()
                    while len(ref) < 3:
                        ref.append(-1)
                    ref_atoms[new_neighbor_node] = ref.copy()

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.append( new_neighbor_node )
        # update the list of most recently visited nodes
        last_nodes_visited = set(neighboring_nodes.copy())

    return nodes_visited, ref_atoms

#other attempts(ignore them)

def old_compute_ref_atoms(molecule):
    '''
    Computes reference atoms for computing internal coordinates using a depth-first algorithm. Assumes the molecule is ordered according to an atom ranking.
        Returns: ref_atoms (list of lists of three ints): reference atoms. The indexes at each triplet are respectively the distance, angle and torsion references.
    '''

    ref_atoms = [ [] for _ in range(molecule.GetNumAtoms()) ]
    
    A = Chem.GetAdjacencyMatrix(molecule)
    n_nodes = A.shape[0]

    highest_ranked_neighbor = []
    for node in range(n_nodes):
        highest_ranked_neighbor.append(np.min([int(i) for i in np.nonzero(A[node, :])[0]])) 
        
    ref_atoms[0] = [None, None, None]
    ref_atoms[1] = [0, None, None]
    if n_nodes > 2:
        r3 = highest_ranked_neighbor[2]
        ref_atoms[2] = [r3, highest_ranked_neighbor[r3], None]
    if n_nodes > 3:
        for i_node in range(3, n_nodes):
            r3 = highest_ranked_neighbor[i_node]
            r2 = highest_ranked_neighbor[r3]
            possible_r1 = [int(i) for i in np.nonzero(A[r2, :])[0] if i!=r3 and i<i_node]
            if possible_r1:
                r1 = np.min(possible_r1)
            else:
                print("The graph-transversing algorithm found an exception computing reference atoms.")
                possible_r1 = [int(i) for i in range(i_node) if i!=r3 and i!=r2]
                r1 = np.min(possible_r1)
            
            ref_atoms[i_node] = [r3,r2,r1]

    return ref_atoms

def ruled_compute_ref_atoms(molecule):
    '''
    Computes reference atoms for computing internal coordinates using a depth-first algorithm. Assumes the molecule is ordered according to an atom ranking.
        Returns: ref_atoms (list of lists of three ints): reference atoms. The indexes at each triplet are respectively the distance, angle and torsion references.
    '''
    # Is it efficient to do all of this while we transverse the graph with the dfs?
    #change lists wih nonempty for neighbours

    ref_atoms = [ [] for _ in range(molecule.GetNumAtoms()) ]
    A = Chem.GetAdjacencyMatrix(molecule)
    n_nodes = A.shape[0]
    non_terminal_nodes = [int(i_node) for i_node in range(n_nodes) if np.sum(A[i_node])>1]
    A_nt = A[non_terminal_nodes,:][:,non_terminal_nodes]

    neighbors_nt = []
    for i_node in range(A_nt.shape[0]):
        neighbors_nt.append([int(i) for i in np.nonzero(A_nt[i_node, :])[0]])
    node_order_nt = dfs(neighbors_nt)
    node_order = []
    for i_node in range(A_nt.shape[0]):
        node_order.append(non_terminal_nodes[node_order_nt[i_node]])
    for i_node in range(n_nodes):
        if i_node not in node_order:
            node_order.append(i_node)

    #sort and re-compute quantities in the new order
    molecule = Chem.RenumberAtoms(molecule, node_order)
    A = Chem.GetAdjacencyMatrix(molecule) #recompute in the new order
    neighbors = []
    for i_node in range(n_nodes):
        neighbors.append([int(i) for i in np.nonzero(A[i_node, :])[0]])

    parent = [None, 0]
    for i_node in range(2, n_nodes):
        placed_neighbours = [int(i) for i in np.nonzero(A[i_node, :])[0] if i<i_node]
        parent.append(min(placed_neighbours)) #both min and max are legit. 
    
    ref_atoms[0] = [None, None, None]
    ref_atoms[1] = [0, None, None]
    if n_nodes > 2:
        if 0 in [int(i) for i in np.nonzero(A[2, :])[0]]:
            ref_atoms[2] = [0, 1, None]
        else:
            ref_atoms[2] = [1, 0, None]

    if n_nodes > 3:    
        for i_node in range(3, n_nodes):
            success = False
            r3 = parent[i_node]
            if r3 == 0:
                r3_placed_neighbours = [int(i) for i in np.nonzero(A[r3, :])[0] if i<i_node]
                r3_placed_neighbours.sort()
                if not r3_placed_neighbours:
                    r3_placed_neighbours = []
                for r2_candidate in r3_placed_neighbours:
                    r2_placed_neighbours = [int(i) for i in np.nonzero(A[r2_candidate, :])[0] if i<i_node and i != r3]
                    if r2_placed_neighbours:
                        r1 = min(r2_placed_neighbours)
                        r2 = r2_candidate
                        success = True
                        break
            else:
                r2 = parent[r3]
                if r2 == 0:
                    r2_placed_neighbours = [int(i) for i in np.nonzero(A[r2, :])[0] if i<i_node and i != r3]
                    if r2_placed_neighbours:
                        r1 = min(r2_placed_neighbours)
                        success = True
                    if not success:
                        #try other r3s
                        r3_candidates = [int(i) for i in np.nonzero(A[i_node, :])[0] if i<i_node]
                        r3_candidates.sort()
                        if not r3_candidates:
                            r3_candidates = []
                        for r3_candidate in r3_candidates:
                            r2_candidates = [int(i) for i in np.nonzero(A[r3_candidate, :])[0] if i<i_node]
                            r2_candidates.sort()
                            if not r2_candidates:
                                r2_candidates = []
                            for r2_candidate in r2_candidates:
                                r1_candidates = [int(i) for i in np.nonzero(A[r2_candidate, :])[0] if i<i_node and i != r3]
                                if r1_candidates:
                                    r3 = r3_candidate
                                    r2 = r2_candidate
                                    r1 = min(r1_candidates)
                                    success = True
                                    
                            if success:
                                break
                else:
                    r1 = parent[r2]
                    success = True
            if success:
                ref_atoms[i_node] = [r3, r2, r1]
            else:
                print("The graph transversing algorithm found an exception computing reference atoms.")
                r3 = min([int(i) for i in np.nonzero(A[i_node, :])[0] if i < i_node])
                r2 = min([int(i) for i in np.nonzero(A[r3, :])[0] if i < i_node])
                r1 = min([int(i) for i in range(i_node) if i!=r3 and i!=r2])
                ref_atoms[i_node] = [r3, r2, r1]

    return ref_atoms

 
"""
    #old parent-based implmentation of compute_ref_atoms_sorted_molecule:
    parent = [None, 0]
    for i_node in range(2, n_nodes):
        placed_neighbours = [int(i) for i in np.nonzero(A[i_node, :])[0] if i<i_node]
        parent.append(min(placed_neighbours)) #both min and max are legit. 
    
    if n_nodes > 3:    
        for i_node in range(3, n_nodes):
            success = False
            r3 = parent[i_node]
            if r3 == 0:
                r3_placed_neighbours = [int(i) for i in np.nonzero(A[r3, :])[0] if i<i_node]
                r3_placed_neighbours.sort()
                if not r3_placed_neighbours:
                    r3_placed_neighbours = []
                for r2_candidate in r3_placed_neighbours:
                    r2_placed_neighbours = [int(i) for i in np.nonzero(A[r2_candidate, :])[0] if i<i_node and i != r3]
                    if r2_placed_neighbours:
                        r1 = min(r2_placed_neighbours)
                        r2 = r2_candidate
                        success = True
                        break
            else:
                r2 = parent[r3]
                if r2 == 0:
                    r2_placed_neighbours = [int(i) for i in np.nonzero(A[r2, :])[0] if i<i_node and i != r3]
                    if r2_placed_neighbours:
                        r1 = min(r2_placed_neighbours)
                        success = True
                    if not success:
                        #try other r3s
                        r3_candidates = [int(i) for i in np.nonzero(A[i_node, :])[0] if i<i_node]
                        r3_candidates.sort()
                        if not r3_candidates:
                            r3_candidates = []
                        for r3_candidate in r3_candidates:
                            r2_candidates = [int(i) for i in np.nonzero(A[r3_candidate, :])[0] if i<i_node]
                            r2_candidates.sort()
                            if not r2_candidates:
                                r2_candidates = []
                            for r2_candidate in r2_candidates:
                                r1_candidates = [int(i) for i in np.nonzero(A[r2_candidate, :])[0] if i<i_node and i != r3_candidate]
                                if r1_candidates:
                                    r3 = r3_candidate
                                    r2 = r2_candidate
                                    r1 = min(r1_candidates)
                                    success = True
                                    
                            if success:
                                break
                else:
                    r1 = parent[r2]
                    success = True
    
            if success:
                ref_atoms[i_node] = [r3, r2, r1]
            else:
                print("The graph transversing algorithm found an exception computing reference atoms.")
                r3 = min([int(i) for i in np.nonzero(A[i_node, :])[0] if i < i_node])
                r2 = min([int(i) for i in np.nonzero(A[r3, :])[0] if i < i_node])
                r1 = min([int(i) for i in range(i_node) if i!=r3 and i!=r2])
                ref_atoms[i_node] = [r3, r2, r1]
"""


