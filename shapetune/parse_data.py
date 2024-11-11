"""
This module contains functions for parsing the data from the input files.
The input files include the sequence identity, the m1psi SHAPE reactivity data for the luciferase and erythropoietein (HEPO) sequences.
The module also contains initial thermodynamic parameters for the RNA folding model.
"""

import dataclasses as dataclass
import os

import pandas as pd

############################################################
####### parse the m1psi SHAPE reactivity data   ############
############################################################


@dataclass.dataclass
class SHAPE_DATA:
    tsv_list = [f for f in os.listdir("data") if f.endswith(".tsv")]
    NAMES = [f.split(".")[0] for f in tsv_list]
    print(NAMES)
    RAW_MOU = {
        name: pd.read_csv(f"data/{f}", sep="\t")
        for f, name in zip(tsv_list, NAMES)
        if name[:3] == "moU"
    }
    RAW_MPU = {
        name: pd.read_csv(f"data/{f}", sep="\t")
        for f, name in zip(tsv_list, NAMES)
        if name[:3] == "mpU"
    }
    RAW_U = {
        name: pd.read_csv(f"data/{f}", sep="\t")
        for f, name in zip(tsv_list, NAMES)
        if name[:1] == "U"
    }

    def __post_init__(self) -> None:
        # concatenate the "Sequence" column for each dataframe
        self.SEQUENCES_MOU = {
            name: "".join(df["Sequence"].values) for name, df in self.RAW_MOU.items()
        }
        self.SEQUENCES_MPU = {
            name: "".join(df["Sequence"].values) for name, df in self.RAW_MPU.items()
        }
        self.SEQUENCES_U = {
            name: "".join(df["Sequence"].values) for name, df in self.RAW_U.items()
        }
        self.REACTIVITY_MOU = {
            name: df["Normalized Reactivity"].values.tolist()
            for name, df in self.RAW_MOU.items()
        }
        self.REACTIVITY_MPU = {
            name: df["Normalized Reactivity"].values.tolist()
            for name, df in self.RAW_MPU.items()
        }
        self.REACTIVITY_U = {
            name: df["Normalized Reactivity"].values.tolist()
            for name, df in self.RAW_U.items()
        }
