import pandas as pd
import numpy as np
import linearpartition as lp
from numpy_ml.neural_nets.optimizers import Adam
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

num_threads = os.cpu_count()

logging.basicConfig(filename='history.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

psi_shape = pd.read_csv('n1mpsi.csv').fillna(float('-inf'))
psi_shape_num = psi_shape.iloc[1:].apply(pd.to_numeric, errors='coerce')
psi_shape_num = psi_shape_num.rename(columns = psi_shape.iloc[0])
#parse the sequence data
full_seq = pd.read_csv('seq_identity.csv')
SEQUENCES = full_seq.set_index('Name')['Sequence']
NAMES = psi_shape.iloc[0].tolist()

psi_shape_dict = dict()
for name in NAMES:
    if name[0] == 'L':
        psi_shape_dict[name] = psi_shape_num[name].values[47:1697]
    else:
        psi_shape_dict[name] = psi_shape_num[name].values[47:626]

PSI_SHAPE2PROB_SIG = dict()
for name in NAMES:
    PSI_SHAPE2PROB_SIG[name] =  1/(np.exp((psi_shape_dict[name] - 2)) + 1)

SYM_4 = [(i, j) for i in range(4) for j in range(i, 4)]

M1PSI = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -52, -70, 0],
    [0, 0, 0, 0, 0, -89, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, -52, -89, 0, 0, -154, -107, 0],
    [0, -70, -1, 0, 0, -107, -112, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype = np.float64)
M1PSI_TERM = np.array([31.0], dtype = np.float64)


def shape_diff(
        name: str, 
        stack_x: np.array,
):
    global SEQUENCES, PSI_SHAPE2PROB_SIG
    if name not in SEQUENCES:
        raise KeyError(f"Name {name} not found in SEQUENCES")
    seq = SEQUENCES[name]
    _, _, probvec = lp.partition(seq, update_stack=stack_x)
    return np.linalg.norm(PSI_SHAPE2PROB_SIG[name] - probvec) ** 2

@lru_cache(maxsize = 1024)
def washi_diff(update: tuple, executor: ThreadPoolExecutor):
    update= np.array(update)
    gc = update[:8].reshape(2, 4)
    psi2_upper = update[8:18]
    psi2 = np.zeros((4, 4), dtype=np.float64)
    global SYM_4
    for idx, (i, j) in enumerate(SYM_4):
        psi2[i, j] = psi2_upper[idx]
        if i != j:
            psi2[j, i] = psi2_upper[idx]
    stack_x = np.zeros((8, 8), dtype = np.float64)
    stack_x[1:3, 3:7] = gc
    stack_x[3:7, 1:3] = gc.T
    stack_x[3:7, 3:7] = psi2

    global M1PSI, M1PSI_TERM
    diff_with_dutta = np.linalg.norm(
        stack_x - M1PSI 
    ) ** 2 

    global NAMES
    futures = [executor.submit(shape_diff, name, stack_x) for name in NAMES]
    results = [future.result() for future in as_completed(futures)]
    prob_diff = sum(results)

    return prob_diff + diff_with_dutta * 0.01

def num_gradient(func: callable, params: tuple, 
                 executor: ThreadPoolExecutor,
                 epsilon = 1e-6, 
                 max_norm = 100):
    def calc_grad(i, func = func, params = params, epsilon = epsilon):
        origin = func(params, executor)
        params_copy = params[:i] + (params[i] + epsilon,) + params[i+1:]
        plus = func(params_copy, executor)
        return (plus - origin) / epsilon
    futures = [executor.submit(calc_grad, i) for i in range(len(params))]
    results = [future.result() for future in as_completed(futures)]
    grads = np.array(results)
    if np.linalg.norm(grads) > max_norm:
        grads = grads / np.linalg.norm(grads) * max_norm
    return grads

if __name__ == "__main__":
    seed = 902
    np.random.seed(seed = seed)
    init = np.array(
        [0, 0, -52, -70,
         0, 0, -89, -1, 
         0, 0,  0,    0,
            0,  0,    0,
             -154, -107, 
                   -112]
    , dtype = np.float64)
    noise = 10 * np.random.randn(init.size)
    init += noise
    name = "stack"

    optimizer = Adam(lr=0.30)
    num_iters = 1000

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_iters):
            logging.info(f"Beginning Loss Function Calc for Iteration {i}:")
            loss = washi_diff(tuple(init), executor)
            grads = num_gradient(washi_diff, tuple(init), executor)
            init = optimizer.update(init, grads, name, cur_loss = loss)

            logging.info(f"Iteration {i} loss: {loss:.4f}")

            total_grad_norm = np.linalg.norm(grads)

            logging.info(f"Total gradient norm: {total_grad_norm:.4f}")

            form_gc = "\n".join([",".join([f"{p:.4f}" for p in row]) for row in init[:8].reshape(2, 4)])
            psi_up = init[8:18]
            psi = np.zeros((4, 4), dtype=np.float64)
            for idx, (i, j) in enumerate(SYM_4):
                psi[i, j] = psi_up[idx]
                if i != j:
                    psi[j, i] = psi_up[idx]
            form_psi2 = "\n".join([",".join([f"{p:.4f}" for p in row]) for row in psi])

            logging.info(f"Updated parameters: \n {form_gc}")
            logging.info(f"Updated Psi-Psi parameters: \n  {form_psi2}")
            logging.info("--------------------")