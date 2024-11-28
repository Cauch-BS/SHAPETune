import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import logging
import sys

import jax.numpy as jnp
import tqdm

from shapetune._spartition import CONSTANTS, partition

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s - %(name)s - %(module)s - %(lineno)s",
    stream=sys.stderr,
)

# read from vienna_efe.txt
vienna_efe = json.load(open("benchmarks/turner.json", "r"))
sequences = list(vienna_efe.keys())

cannon_efes = []
cannon_spps = []
batch_size = 125
for i in tqdm.tqdm(range(0, len(sequences), batch_size)):
    cannon_efe = partition(sequences[i : i + batch_size], CONSTANTS.STACKING_PAIRS)
    efes = cannon_efe[1].tolist()
    spp = (
        (cannon_efe[0] + jnp.transpose(cannon_efe[0], axes=(0, 2, 1)))
        .sum(axis=(1, 2))
        .tolist()
    )
    cannon_efes.extend(efes)
    cannon_spps.extend(spp)
    logging.info(f"Processed {i + batch_size} sequences")
    logging.info(f"EFE: \n {efes}")
    logging.info(f"SPP: \n {spp}")

spartition_efe = dict(zip(sequences, zip(cannon_efes, cannon_spps)))
with open("benchmarks/spartition.json", "w") as f:
    json.dump(spartition_efe, f)
