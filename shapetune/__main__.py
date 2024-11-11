"""
Main Code for optimization of thermodynamic parameters.
Currently, optimization based on minimization of Jensen-Shannon divergence is implemented.
"""

# from shapetune.parse_data import SHAPE, parse_stacks

# psi_shape_dict = dict()

# for name in SHAPE.NAMES:
#     if name[0] == "L":
#         psi_shape_dict[name] = SHAPE.psi_shape_num[name].values
#     else:
#         psi_shape_dict[name] = SHAPE.psi_shape_num[name].values

# PSI_SHAPE2PROB_SIG = dict()
# for name in SHAPE.NAMES:
#     PSI_SHAPE2PROB_SIG[name] = 1 / (np.exp((psi_shape_dict[name] - 2)) + 1)


# M1PSI = parse_stacks("dutta")
# M1PSI_TERM = np.array([31.0], dtype=np.float64)


# def shape_diff(
#     name: str,
#     stack_x: np.ndarray,
# ):
#     if name not in SHAPE.SEQUENCES:
#         raise KeyError(f"Name {name} not found in SEQUENCES")
#     seq = SHAPE.SEQUENCES[name]
#     # folded = RNA.fold_compound(seq)
#     # partition_seq = folded.pf()
#     # bpmtx = folded.bpp()
#     probvec = np.zeros(len(seq), dtype=np.float64)
#     return np.linalg.norm(PSI_SHAPE2PROB_SIG[name] - probvec) ** 2


# @lru_cache(maxsize=1024)
# def washi_diff(update: tuple, executor: ThreadPoolExecutor, names: tuple):
#     update_arr = np.array(update)
#     gc = update_arr[:8].reshape(2, 4)
#     psi2_upper = update_arr[8:18]
#     psi2 = np.zeros((4, 4), dtype=np.float64)
#     for inner_idx, (i, j) in enumerate(SHAPE.SYM_4):
#         psi2[i, j] = psi2_upper[inner_idx]
#         if i != j:
#             psi2[j, i] = psi2_upper[inner_idx]
#     stack_x = np.zeros((8, 8), dtype=np.float64)
#     stack_x[1:3, 3:7] = gc
#     stack_x[3:7, 1:3] = gc.T
#     stack_x[3:7, 3:7] = psi2

#     diff_with_dutta = np.linalg.norm(stack_x - M1PSI) ** 2

#     futures = [executor.submit(shape_diff, name, stack_x) for name in names]
#     results = [future.result() for future in as_completed(futures)]
#     prob_diff = sum(results)

#     return prob_diff + diff_with_dutta * 0.01


# def num_gradient(
#     func: Callable,
#     params: tuple,
#     names: tuple,
#     executor: ThreadPoolExecutor,
#     epsilon=1e-6,
#     max_norm=10.0,
# ):
#     def calc_grad(i, func=func, names=names, params=params, epsilon=epsilon):
#         origin = func(params, executor, names)
#         params_copy = params[:i] + (params[i] + epsilon,) + params[i + 1 :]
#         plus = func(params_copy, executor, names)
#         return (plus - origin) / epsilon

#     futures = [executor.submit(calc_grad, i) for i in range(len(params))]
#     results = [future.result() for future in as_completed(futures)]
#     grads = np.array(results)
#     if np.linalg.norm(grads) > max_norm:
#         grads = grads / np.linalg.norm(grads) * max_norm
#     return grads


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         description="Run optimization for the Washietl model"
#     )
#     parser.add_argument(
#         "--p", type=int, default=1, help="Number of threads to use for optimization"
#     )
#     parser.add_argument(
#         "--seed", type=int, default=922, help="Random seed (default: 904)"
#     )
#     parser.add_argument(
#         "--lr", type=float, default=0.05, help="Learning rate (default: 0.05)"
#     )
#     parser.add_argument(
#         "--log",
#         type=str,
#         default="history_.log",
#         help="Log file name (default: history_.log)",
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_arguments()
#     logging.basicConfig(
#         filename=args.log,
#         level=logging.INFO,
#         format="%(asctime)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     with ThreadPoolExecutor(max_workers=args.p) as executor:
#         pass
