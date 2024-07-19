import torch
import pandas as pd
import numpy as np
import linearpartition as lp
from torch.optim import Adam
import logging

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

def symmetrize(matrix):
    return (matrix + matrix.T) / 2

M1PSI = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -52, -70, 0],
    [0, 0, 0, 0, 0, -89, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, -52, -89, 0, 0, -154, -107, 0],
    [0, -70, -1, 0, 0, -107, -112, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype = torch.float64)
M1PSI_TERM = torch.tensor([31.0])

def washi_diff(gc: torch.Tensor,
               psi2: torch.Tensor,
               term: torch.Tensor):
    stack_x = torch.zeros(8, 8, dtype = torch.float64)
    stack_x[1:3, 3:7] = gc
    stack_x[3:7, 1:3] = gc.T
    stack_x[3:7, 3:7] = psi2
    terminal_x = term

    global M1PSI, M1PSI_TERM
    diff_with_dutta = torch.linalg.matrix_norm(
        stack_x - M1PSI 
    ) + (terminal_x - M1PSI_TERM) ** 2

    prob_diff = 0.0

    global SEQUENCES, NAMES, PSI_SHAPE2PROB_SIG
    for name in NAMES:
        seq = SEQUENCES[name]
        bpmtx, fe = lp.partition(seq, 
                        update_stack = stack_x, 
                        update_terminal = terminal_x)
        bpp = pd.DataFrame(bpmtx)
        
        bpp_vector = np.array([
                (bpp['prob'][bpp['i'] == i].sum() + 
                bpp['prob'][bpp['j'] == i].sum()) 
            for i in range(0, len(seq))
        ])

        prob_diff += np.linalg.norm(
            PSI_SHAPE2PROB_SIG[name] - bpp_vector
        )
    
    logging.info(f"Step From Dutta: {diff_with_dutta.item() : .4f}")
    logging.info(f"Step From Prob: {prob_diff : .4f}")
    return prob_diff



if __name__ == "__main__":

    seed = 50
    torch.manual_seed(seed = seed)

    init_gc = torch.tensor([
        [0, 0, -52, -70],
        [0, 0, -89,  -1]
    ], dtype= torch.float64)
    rand_gc = 10 * torch.randn(2, 4)
    init_gc += rand_gc
    init_gc.requires_grad_(True)

    init_psi2 = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, -154, -107],
        [0, 0, -107, -112]
    ], dtype = torch.float64)
    rand_psi2 = 10 * torch.rand(4, 4)
    init_psi2 += rand_psi2
    init_psi2.requires_grad_(True)

    init_term = torch.tensor([31.0], dtype=torch.float64)
    rand_term = 10 * torch.randn(1)
    init_term += rand_term
    init_term.requires_grad_(True)

    optimizer = Adam([{'params': init_gc}, {'params': init_psi2}, {'params': init_term}], lr = 0.5)
    num_iters = 1000

    for i in range(num_iters):

        optimizer.zero_grad()
        loss = washi_diff(init_gc, init_psi2, init_term)
        loss.backward()
        
        # Print gradients
        print(f"Iteration {i} loss: {loss.item(): .4f}")
        logging.info(f"Iteration {i} loss: {loss.item() : .4f}")

        optimizer.step()
        with torch.no_grad():
            init_psi2.data = symmetrize(init_psi2.data)

        form_gc = "\n".join([",".join([f"{p:.4f}" for p in row]) for row in init_gc.detach().tolist()])
        form_psi2 = "\n".join([",".join([f"{p:.4f}" for p in row]) for row in init_psi2.detach().tolist()])
        form_term = f"{init_term.item():.4f}"
        print(f"Updated GC parameters: \n {form_gc}")
        print(f"Updated Psi-Psi parameters: \n {form_psi2}")
        print(f"Updated Terminal parameters: {form_term}")
        print("--------------------")
        logging.info(f"Updated parameters: \n {form_gc}")
        logging.info(f"Updated Psi-Psi parameters: \n  {form_psi2}")
        logging.info(f"Updated Terminal parameters: {form_term}")
        logging.info("--------------------")