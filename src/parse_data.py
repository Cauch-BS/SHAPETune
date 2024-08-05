"""
This module contains functions for parsing the data from the input files.
The input files include the sequence identity, the m1psi SHAPE reactivity data for the luciferase and erythropoietein (HEPO) sequences.
The module also contains initial thermodynamic parameters for the RNA folding model.
"""

import pandas as pd
import numpy as np

############################################################
####### parse the m1psi SHAPE reactivity data   ############
############################################################

UTR_5 = "AATATAAGAGCCACC".replace("T", "U")
UTR_3 = "TGATAATAG".replace("T", "U")

psi_shape = pd.read_csv("data/n1mpsi.csv").fillna(float("-inf"))
psi_shape_num = psi_shape.iloc[1:].apply(pd.to_numeric, errors="coerce")
psi_shape_num = psi_shape_num.rename(columns=psi_shape.iloc[0])
# parse the sequence data
full_seq = pd.read_csv("data/seq_identity.csv")
SEQUENCES = full_seq.set_index("Name")["Sequence"].apply(lambda x: UTR_5 + x + UTR_3)
NAMES = psi_shape.iloc[0].tolist()
LUCIFERASE = tuple(elem for elem in NAMES if elem[0] == "L")
HEPO = tuple(elem for elem in NAMES if elem[0] != "L")

psi_shape_dict = dict()
for name in NAMES:
    if name[0] == "L":
        psi_shape_dict[name] = psi_shape_num[name].values[32:1706]
    else:
        psi_shape_dict[name] = psi_shape_num[name].values[32:635]

PSI_SHAPE2PROB_SIG = dict()
for name in NAMES:
    PSI_SHAPE2PROB_SIG[name] = 1 / (np.exp((psi_shape_dict[name] - 2)) + 1)

SYM_4 = [(i, j) for i in range(4) for j in range(i, 4)]

############################################################
####### parse the thermodynamic parameters      ############
############################################################


def parse_stacks(filename: str) -> np.array:
    """Parse the stacks from the input file."""
    M1PSI = np.zeros((8, 8), dtype=np.float64)
    line_it = 0
    with open(f"data/m1psi_{filename}.par", "r", encoding="utf-8") as STACKS:
        for line in STACKS:
            data = np.array(list(map(int, line.strip("{} \n").split(","))))
            M1PSI[line_it] = data
            line_it += 1
    return M1PSI
