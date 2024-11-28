import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tqdm
import ViennaRNA as RNA


# find partition of ViennaRNA
def turner(seq):  # type: ignore
    fc = RNA.fold_compound(seq)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    _, efe = fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    spp = (bpp + bpp.T).sum(axis=(0, 1))
    return (efe, spp)


if __name__ == "__main__":
    import json

    # generate 5000 random sequences between 100 and 200 nt
    random.seed(2)
    sequences = [
        "".join(random.choice("ACGU") for _ in range(2500)) for _ in range(5000)
    ]

    print(turner(sequences[0]))

    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(turner, sequences), total=len(sequences)))
        result = dict(zip(sequences, results))

    with open("data/turner.json", "w") as f:
        json.dump(result, f)
