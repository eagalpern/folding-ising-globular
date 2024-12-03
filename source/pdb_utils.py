import prody
import scipy.spatial.distance as sdist
import numpy as np

def get_distance_matrix(pdb_file: str, 
                        chain: str, 
                        method: str = 'CB'
                        ) -> np.array:
    """
    Calculate the distance matrix of the specified atoms in a PDB file.
    
    Parameters:
        pdb_file (str): The path to the PDB file.
        chain (str): The chainID or chainIDs (space separated) of the protein.
        method (str): The method to use for calculating the distance matrix. 
                      Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
                      Other options are 'CA' for using only the CA atom, 
                      and 'minimum' for using the minimum distance between all atoms in each residue.
    
    Returns:
        np.array: The distance matrix of the selected atoms.
    
    Raises:
        IndexError: If the selection of atoms is empty.
    """
    
    structure = prody.parsePDB(pdb_file)
    chain_selection = '' if chain is None else f' and chain {chain}'
    if method == 'CA':
        selection = structure.select('protein and name CA' + chain_selection)
        if len(selection) == 0:
            raise IndexError('Empty selection')
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'CB':
        selection = structure.select('(protein and (name CB) or (resname GLY and name CA))' + chain_selection)
        if len(selection) == 0:
            raise IndexError('Empty selection')
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'minimum':
        selection = structure.select('protein' + chain_selection)
        if len(selection) == 0:
            raise IndexError('Empty selection')
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        resids = selection.getResindices()
        residues = pd.Series(resids).unique()
        selections = np.array([resids == a for a in residues])
        dm = np.zeros((len(residues), len(residues)))
        for i, j in itertools.combinations(range(len(residues)), 2):
            d = distance_matrix[selections[i]][:, selections[j]].min()
            dm[i, j] = d
            dm[j, i] = d
        return dm

    
def eliminate_diagonals(matrix, k):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")

    L = matrix.shape[0]
    # Create a mask for the diagonals to remove
    mask = np.ones((L, L), dtype=bool)
    for i in range(-k, k + 1):
        mask &= ~np.eye(L, dtype=bool, k=i)
    # Apply the mask to the matrix
    result = matrix * mask
    return result




