"""
This module contains functions for pretty printing numpy arrays.
The energy parameter is given as a 1x18 numpy array. The first 8 elements
are the GC pairs, the next 10 elements are the upper triangular elements of
the 4*4 matrix psi2. 
"""
import numpy as np

SYM_4 = [(i, j) for i in range(4) for j in range(i, 4)]
def prettier_energy(energy: np.array) -> str:
    """
    Pretty print the energy parameter. 
    Formats the energy parameter (as 1 dimensional np.array) as a two dimensional array. 
    The output is a string with the GC pairs and the upper triangular elements of the 4*4 matrix psi2. 
    """
    gc = energy[:8].reshape(2, 4)
    psi2_upper = energy[8:18]
    psi2 = np.zeros((4, 4), dtype=np.float64)
    for idx, (i, j) in enumerate(SYM_4):
        psi2[i, j] = psi2_upper[idx]
        if i != j:
            psi2[j, i] = psi2_upper[idx]
    
    format_gc = "\n".join(['\t' + ','.join([f"{gc[i, j]:.2f}" for j in range(4)]) for i in range(2)])
    format_psi2 = "\n".join(['\t' + ','.join([f"{psi2[i, j]:.2f}" for j in range(4)]) for i in range(4)])

    return f"\n GC:\n{format_gc}\n Psi2:\n{format_psi2}"