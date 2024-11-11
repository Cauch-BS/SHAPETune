import json

import numpy as np
import ViennaRNA as RNA
from scipy.sparse import csr_matrix


def load_params(stack_params: dict[str, float]) -> str:
    """
    Convert a dictionary of parameters to a JSON string.
    Sample Input:
    {
        "A11A" : -1.52,
        "11AA" : -1.23,
        "1AA1" : -1.71,
        "C1GA" : -2.10,
        "G1CA" : -2.50,
        "1CAG" : -2.51,
        "1GAC" : -2.35
    },
    """
    assert all(
        len(key) == 4 for key in stack_params.keys()
    ), "Stack key is not 4 characters"
    assert all(
        set(key).issubset("ACGU1") for key in stack_params.keys()
    ), "Stack key contains invalid character"
    params = {
        "modified_base": {
            "unmodified": "U",
            "pairing_partners": ["A"],
            "one_letter_code": "1",
            "fallback": "U",
            "terminal_energies": {"1A": 0.31, "A1": 0.31},
        }
    }
    params["modified_base"]["stacking_energies"] = stack_params
    return json.dumps(params, indent=4)


def pairprob_arr(fold_compound: RNA.fold_compound) -> np.ndarray:
    """
    Calculate a vector of paired probabilities for each nucleotide in the sequence.
    """
    sparse_bpp = csr_matrix(np.array(fold_compound.bpp()))[1:, 1:]
    sparse_bpp += sparse_bpp.T
    pi_array: np.ndarray = sparse_bpp.sum(axis=0)
    return pi_array


def bpp_params(seq: str, params: dict[str, float]) -> np.ndarray:
    """
    Calculate the paired probabilities for each nucleotide in the sequence given
    the sequence as a string and the parameters as a dictionary.
    """
    fc = RNA.fold_compound(seq)
    _ = fc.sc_mod_json(
        json=load_params(params),
        modification_sites=[i for i, char in enumerate(seq, start=1) if char == "U"],
    )
    _ = fc.pf()
    return pairprob_arr(fc)
