# This code was ported from https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
# Summary of Alterations
#   + The code was changed to optimize for *numpy arrays*, not *pytorch tensors*.
#   + Single name variables were changed to more descriptive names.
#   + Additional documentation was given for more clarity.
#   + The code was refactored to use a logger object. This is because evaluation of the lp.partition function
#     is expensive, and we need to keep track of the values and its evaluations
# Access Date: 2024-08-06
# Accessed at Commit: 345bea0
# Written by : @Cauch-BS (Chaebeom Sheen)
"""
This module contains an optimizer for optimization of the energy parameters.
It is based on the L-BFGS algorithm, which is a quasi-Newton optimization algorithm.
The algorithm approximates the inverse Hessian matrix using a limited memory approach.
Unlike the Bayesian Optimization algorithm, the L-BFGS algorithm requires the calculation
of the gradient of the function to optimize. The algorithm is efficient for large-scale
optimization problems, as it does not require the computation of the full Hessian matrix. However,
it is unsuitable for optimization of a large number of parameters, as it has high memory requirements.
This implementation is based on the PyTorch implementation of the L-BFGS algorithm.
"""

import logging
from collections import deque
from typing import Any, Callable, Dict, Tuple

import numpy as np

from .prettier import prettier_energy


def _cubic_interpolate(
    x1: float,
    fn1: float,
    grad1: float,
    x2: float,
    fn2: float,
    grad2: float,
    bounds: list = [],
) -> float:
    """Ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    Code of the most common case: cubic interpolation between two points
    x1, x2 given their corresponding function values (fn1, fn2) and their gradients
    grad1, grad2. Solution in this case (where x2 is the farthest point):
    d1 = grad2 + grad1 - 3 * (fn2 - fn1) / (x2 - x1)
    d2 = sqrt(d1^2 - grad1 * grad2)
    x_min = x2 - (x2 - x1) * (grad2 + d2 - d1) / (grad2 - grad1 + 2 * d2)
    t_new = min(max(x_min, xmin_bound), xmax_bound)
    """
    # generate bounds
    if bounds:
        xmin_bound, xmax_bound = bounds
    elif bounds == []:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # calculate the coefficients of the cubic polynomial
    d1 = grad2 + grad1 - 3 * (fn2 - fn1) / (x2 - x1)
    d2_square = d1**2 - grad1 * grad2
    if d2_square >= 0:
        d2 = np.sqrt(d2_square)
        x_min = xmax_bound - (xmax_bound - xmin_bound) * (grad2 + d2 - d1) / (
            grad2 - grad1 + 2 * d2
        )
        t_new: float = min(max(x_min, xmin_bound), xmax_bound)
        return t_new
    else:
        return (x1 + x2) / 2


def _strong_wolfe(
    obj_fn: Callable[[Tuple[float, ...], float, np.ndarray], Tuple[float, np.ndarray]],
    init: Tuple[float, ...],
    step: float,
    dir: np.ndarray,
    val: float,
    gradv: np.ndarray,
    grd_prod: float,  # dot product between the gradient and the search direction
    logger: logging.Logger,
    armijo: float = 1e-4,
    curv: float = 0.9,
    tol: float = 1e-9,
    max_iter: int = 25,
) -> tuple[float, np.ndarray, float, int]:
    """Ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    Arguments:
        obj_fn: function to minimize
            + if ori_fn is the original function to minimize then obj_fn is a function that takes
              a point and a step size and returns the value of the function at the new point and the gradient of the function at the new point
        init: initial argument of the function
        step: initial step size
        dir: search direction
        val: value of the function at the initial point
        gradv: gradient value of the function at the initial point
        grd_prod: dot product between the gradient and the search direction
        logger: logger object
        armijo: armijo condition parameter
        curv: curvature condition parameter
        tol: tolerance
        max_iter: maximum number of iterations of Line Search
    Returns:
        val_new: the value of the function at the new point
        gradv_new: the gradient value of the function at the new point
        step: the step size that satisfies the strong wolfe conditions
        ls_iter: the number of iterations needed to find the step size
    Explanation:
        The function finds the step size that satisfies the Strong Wolfe conditions
        using cubic interpolation and the Armijo condition. The Armijo condition is:
            1) f(x + step * dir) <= f(x) + armijo * step * < gradv | dir >
        The curvature condition is:
            2) - < dir | gradfn(x + step * dir) > <= - curv < gradv | dir >
        Both the armijo coefficient and the curvature coefficient are in (0, 1).
        It is recommended to use armijo = 1e-4 and curv = 0.9. curv should be much larger than armijo.
        The first condition ensures that the step length decreases the function sufficiently,
        while the second condition ensures that the slope length is sufficiently decreased.
        The first and second condition can be interpretd as respectively providing an
        upper and lower bound on the step size. If we modify 2) as
            3) || < dir | gradfn(x + step * dir) > ||  >= curv || < gradv | dir > ||
        then 1) and 3) form the strong wolfe-conditions, and force the step to lie close to a
        critical point of fn.
    """
    logger.info("Starting Line Search for Point Satisfying Strong Wolfe Conditions...")
    # Added Logger Support (Why? Because obj_fn is expensive to compute. Thus we need to keep track of the values) ~~
    dir_norm = np.abs(dir).max()  # Find the ||L||_inf norm of the direction
    gradv = np.copy(gradv)
    val_new, gradv_new = obj_fn(
        init, step, dir
    )  # Evaluate the function at the new point
    logger.info(f"\t Initial Loss: {val}, New Loss at Iteration 1: {val_new}")
    logger.info(
        f"\t Initial Gradient Size: {np.linalg.norm(gradv)}, New Gradient Size at Iteration 1: {np.linalg.norm(gradv_new)}"
    )
    ls_iter = 1  # First Line Search iteration
    grd_prod_new = np.dot(gradv_new, dir)

    # bracket an interval containing a point that satisfies the strong wolfe conditions
    step_prev, val_prev, gradv_prev, grd_prod_prev = 0.0, val, gradv, grd_prod
    done = False

    while ls_iter <= max_iter:
        # check the armijo condition
        if (val_new > val + armijo * step * grd_prod) or (
            ls_iter > 2 and val_new >= val_prev
        ):
            bracket = [step_prev, step]
            bracket_val = [val_prev, val_new]
            bracket_grad = [gradv_prev, np.copy(gradv_new)]
            bracket_grd_prod = [grd_prod_prev, grd_prod_new]
            logging.info("\t Terminating Line Search: Armijo Condition Satisfied.")
            break

        # check the curvature condition
        if np.abs(grd_prod_new) <= -curv * grd_prod:
            bracket = [step]
            bracket_val = [val_new]
            bracket_grad = [gradv_new]
            done = True
            logging.info("\t Terminating Line Search: Curvature Condition Satisfied.")
            break

        # check positive condition
        if grd_prod_new >= 0:
            bracket = [step_prev, step]
            bracket_val = [val_prev, val_new]
            bracket_grad = [gradv_prev, np.copy(gradv_new)]
            bracket_grd_prod = [grd_prod_prev, grd_prod_new]
            logging.info("\t Terminating Line Search: Positive Condition Satisfied.")
            break

        # cubic interpolation
        min_step = step + 0.01 * (step - step_prev)
        max_step = step * 10
        temp = step
        step = _cubic_interpolate(
            step_prev,
            val_prev,
            grd_prod_prev,
            step,
            val_new,
            grd_prod_new,
            bounds=[min_step, max_step],
        )

        # evaluate the function at the new point
        step_prev = temp
        val_prev = val_new
        gradv_prev = np.copy(gradv_new)
        grd_prod_prev = grd_prod_new
        val_new, gradv_new = obj_fn(init, step, dir)
        grd_prod_new = np.dot(gradv_new, dir)
        logger.info(f"\t New Loss at iteration {ls_iter}: {val_new}")
        logger.info(
            f"\t New Gradient Size at iteration {ls_iter}: {np.linalg.norm(gradv_new)}"
        )
        ls_iter += 1

    # Reached the maximum number of iterations
    if ls_iter > max_iter:
        bracket = [0, step]
        bracket_val = [val, val_new]
        bracket_grad = [gradv, gradv_new]

    # Zoom phase: We now have an interval containing a point that satisfies the strong wolfe conditions,
    # or else, we have a point that satisfies the curvature condition. We now refine the interval
    # unitl we find the exact point that satisfies the strong wolfe conditions.
    insuf_progress = False
    low_pos, high_pos = (0, -1) if bracket_val[0] < bracket_val[-1] else (-1, 0)
    while (
        not done and ls_iter <= max_iter
    ):  # if the curvature condition is not satisfied
        # if line-search bracket is too small, then we are not making progress
        if np.abs(bracket[1] - bracket[0]) * dir_norm < tol:
            break

        # compute new step via cubic interpolation
        step = _cubic_interpolate(
            bracket[0],
            bracket_val[0],
            bracket_grd_prod[0],
            bracket[1],
            bracket_val[1],
            bracket_grd_prod[1],
        )

        # test that we are making sufficient progress
        # if step is too close to the boundary, then we are not making progress
        # and if
        #   + we have not made sufficient progres in the last iteration
        #   + step is at the boundary
        # we will move step to a position which is 0.1 * μ(bracket) away from the boundary
        # where μ is the measue of the bracket (i.e. the length)
        eps = 0.1 * (bracket[high_pos] - bracket[low_pos])
        if min(bracket[high_pos] - step, step - bracket[low_pos]) < eps:
            if insuf_progress or step >= bracket[high_pos] or step <= bracket[low_pos]:
                if np.abs(step - bracket[high_pos]) < np.abs(step - bracket[low_pos]):
                    step = bracket[high_pos] - eps
                else:
                    step = bracket[low_pos] + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # evaluate the function at the new point
        val_new, gradv_new = obj_fn(init, step, dir)
        ls_iter += 1
        grd_prod_new = np.dot(gradv_new, dir)
        logger.info(f"\t New Loss at iteration {ls_iter}: {val_new}")
        logger.info(
            f"\t New Gradient Size at iteration {ls_iter}: {np.linalg.norm(gradv_new)}"
        )

        if val_new > val + armijo * step * grd_prod or val_new >= bracket_val[low_pos]:
            # the armijo condition is not satisfied
            # or the new value is greater than the lowest value in the bracket
            bracket[high_pos] = step
            bracket_val[high_pos] = val_new
            bracket_grad[high_pos] = np.copy(gradv_new)
            bracket_grd_prod[high_pos] = grd_prod_new
            low_pos, high_pos = (0, 1) if bracket_val[0] < bracket_val[1] else (1, 0)
        else:
            if np.abs(grd_prod_new) <= -curv * grd_prod:
                # the Wolfe conditions are satisfied
                done = True
            elif grd_prod_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_val[high_pos] = bracket_val[low_pos]
                bracket_grad[high_pos] = bracket_grad[low_pos]
                bracket_grd_prod[high_pos] = bracket_grd_prod[low_pos]
            # new point becomes low point
            bracket[low_pos] = step
            bracket_val[low_pos] = val_new
            bracket_grad[low_pos] = np.copy(gradv_new)
            bracket_grd_prod[low_pos] = grd_prod_new

    if ls_iter > max_iter:
        logger.info("\t Terminating Line Search: Maximum Iterations Reached.")

    step = bracket[low_pos]
    val_new = bracket_val[low_pos]
    gradv_new = bracket_grad[low_pos]
    return val_new, gradv_new, step, ls_iter


class LBFGS:
    """Implements the L-BFGS optimization algorithm.
    Arguments:
        params: tuple of parameters to optimize
        func: function to optimize, with a tuple as input and float as output
        grad: gradient of the function, with a tuple as input and numpy array as output
        lr: learning rate (default: 1.0)
        max_iter: maximum number of iterations (default: 20)
        max_eval: maximum number of evaluations (default: max_iter * 5 // 4)
        tolerance_grad: tolerance for the gradient (default: 1e-7)
        tolerance_change: tolerance for the change in the function value (default: 1e-9)
        history_size: size of the history (default: 10)
        line_search_fn: line search function (default: None)
    Methods:
        optimize: performs optimization based on the L-BFGS algorithm
            Explanation:
                The L-BFGS algorithm is a quasi-Newton optimization algorithm that approximates the inverse Hessian matrix.
                The algorithm is based on the BFGS algorithm, but uses a limited memory approach to store the history of the
                gradients and updates. The algorithm is efficient for large-scale optimization problems, as it does not require
                the computation of the full Hessian matrix. The algorithm is also robust to noisy gradients, as it uses an
                approximation of the Hessian matrix to update the parameters. Robustness against non-convex functions is also
                a feature of the algorithm through the use of the Wolfe conditions. The algorithm is widely used in machine learning.
                For a detailed explanation of the algorithm, see Chapter 6. Quasi-Newton Methods in the book Numerical Optimization (2nd ed.)
                by Nocedal and Wright (2006).
    """

    def __init__(
        self,
        params: tuple,
        func: Callable[[Tuple[float, ...]], float],
        grad: Callable[[Tuple[float, ...]], np.ndarray],
        logger: logging.Logger,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int = 0,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 10,
        line_search_fn: str = "",
    ) -> None:
        self.params = params
        self.func = func
        self.grad = grad
        self.lr = lr
        self.logger = logger
        self.max_iter = max_iter
        self.max_eval = max_eval if max_eval != 0 else max_iter * 25
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self._numel_cache = None
        self.state: Dict[str, Any] = {
            "func_evals": 0,
            "grad_evals": 0,
            "n_iter": 0,
            "step": lr,
            "dir": None,
            "old_dirs": deque(maxlen=history_size),
            "old_stps": deque(maxlen=history_size),
            "ro": deque(
                maxlen=history_size
            ),  # ro is the ρ value in the L-BFGS algorithm
            "hess_diag": 1,
            "al": [None] * history_size,  # al is the α value in the L-BFGS algorithm
        }

    @staticmethod
    def _add_grad(
        params: Tuple[float, ...], step_size: float, update: np.ndarray
    ) -> Tuple[float, ...]:
        assert len(params) == len(update), "Parameter and Gradient Dimension Mismatch"
        new_par: Tuple[float, ...] = tuple(
            p + step_size * u for p, u in zip(params, update)
        )
        return new_par

    def _directional_evaluate(self, init, step, direction) -> tuple:  # type: ignore
        """Evaluate the function at the new point"""
        updated = self._add_grad(init, step, direction)
        logging.info(
            f"\t Evaluating Function at New Point: {prettier_energy(np.array(updated))}"
        )
        new_val = self.func(updated)
        self.state["func_evals"] += 1
        new_grad = self.grad(updated)
        self.state["grad_evals"] += 1
        return new_val, new_grad

    def optimize(self) -> float:
        """Performs a single optimization step.
        Args:
            prev: result of the previous optimization step
        """
        loss = orig_loss = self.func(self.params)
        self.state["func_evals"] += 1
        gradient = self.grad(self.params)
        self.state["grad_evals"] += 1
        opt_cond = np.abs(gradient).max() <= self.tolerance_grad
        if opt_cond:
            self.logger.info("Terminating Optimization: Gradient goes to zero")
            return orig_loss

        dir = self.state["dir"]
        step = self.state["step"]
        old_dirs = self.state["old_dirs"]
        old_stps = self.state["old_stps"]
        ro = self.state["ro"]
        hess_diag = self.state[
            "hess_diag"
        ]  # hess_diag is the diagonal of the Hessian matrix
        prev_grad = self.state.get("prev_grad", None)
        prev_loss = self.state.get("prev_loss", None)

        while self.state["n_iter"] < self.max_iter:
            self.state["n_iter"] += 1
            ############################################################
            #               Compute the search direction               #
            ############################################################
            if self.state["n_iter"] == 1:
                dir = -gradient
            else:
                # Do a quasi-Newton update
                curv = gradient - prev_grad
                stepdir = step * dir
                curv_stepdir = np.dot(curv, stepdir)
                if curv_stepdir > 1e-10:
                    old_dirs.append(curv)
                    old_stps.append(stepdir)
                    ro.append(1.0 / curv_stepdir)
                    # update the inverse Hessian approximation
                    hess_diag = curv_stepdir / np.dot(curv, curv)

                # compute the approximate L-BFGS inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)
                al = self.state["al"]
                q = -gradient  # q is initialized as the negative gradient
                for i in range(num_old - 1, -1, -1):
                    al[i] = ro[i] * np.dot(old_stps[i], q)
                    q -= al[i] * old_dirs[i]

                dir = r = hess_diag * q  # multiply by the initial Hessian approximation
                # r / dir is the search direction
                for i in range(num_old):
                    be_i = ro[i] * np.dot(old_dirs[i], r)
                    r += (al[i] - be_i) * old_stps[i]
            if prev_grad is None:
                prev_grad = np.copy(gradient)
            else:
                prev_grad[:] = gradient

            prev_loss = loss

            ############################################################
            #               Line Search (Length of Step)               #
            ############################################################
            # reset the initial guess for the step size
            if self.state["n_iter"] == 1:
                step = min(1.0, 1.0 / np.abs(gradient).sum()) * self.lr
            else:
                step = self.lr

            # get the directional derivative
            grad_dir = np.dot(gradient, dir)
            if grad_dir > -self.tolerance_change:
                self.logger.info(
                    "Terminating Optimization: Directional Derivative is too small."
                )
                break
            if self.line_search_fn != "":
                if self.line_search_fn != "strong_wolfe":
                    raise NotImplementedError(
                        "Line search function not recognized. Only 'strong_wolfe' is supported."
                    )
                else:
                    init = self.params

                    def obj_func(init, step, dir):  # type: ignore
                        return self._directional_evaluate(init, step, dir)

                    loss, gradient, step, _ = _strong_wolfe(
                        obj_func, init, step, dir, loss, gradient, grad_dir, self.logger
                    )
                self.params = self._add_grad(self.params, step, dir)
                opt_cond = np.abs(gradient).max() <= self.tolerance_grad
            elif self.line_search_fn == "":
                # no line search, simply move with the step size (fixed step size)
                loss, gradient = self._directional_evaluate(self.params, step, dir)
                opt_cond = np.abs(gradient).max() <= self.tolerance_grad

            ############################################################
            #              Check For Termination                       #
            ############################################################
            if self.state["n_iter"] >= self.max_iter:
                self.logger.info(
                    "Terminating L-BFGS Optimization: Maximum Iterations Reached."
                )
                break

            if self.state["func_evals"] >= self.max_eval:
                self.logger.info(
                    "Terminating L-BFGS Optimization: Maximum Evaluations Reached."
                )
                break

            if opt_cond:
                self.logger.info(
                    "Terminating L-BFGS Optimization: Gradient is sufficiently small."
                )
                break

            if np.abs(dir * step).max() <= self.tolerance_change:
                self.logger.info(
                    "Terminating L-BFGS Optimization: Change in parameter value is too small."
                )
                break

            if np.abs(loss - prev_loss) < self.tolerance_change:
                self.logger.info(
                    "Terminating L-BFGS Optimization: Change in function value is too small."
                )
                break

        self.state["dir"] = dir
        self.state["step"] = step
        self.state["old_dirs"] = old_dirs
        self.state["old_stps"] = old_stps
        self.state["ro"] = ro
        self.state["hess_diag"] = hess_diag
        self.state["prev_grad"] = prev_grad
        self.state["prev_loss"] = prev_loss
        self.logger.info(
            f"Optimization Complete. Final Loss: {loss}, Final Gradient: {np.linalg.norm(gradient)}"
        )
        return loss

    def stochastic_optimize(self) -> None:
        """Performs a single stochastic optimization step
        The code below is not from the PyTorch codebase.
        It follows the implementation by Mortiz P, Nishihara R, and Jordan M. from UC Berkeley
        The relevant paper is quoted below in BibTex format:
        @InProceedings{pmlr-v51-moritz16,
          title = 	 {A Linearly-Convergent Stochastic L-BFGS Algorithm},
          author = 	 {Moritz, Philipp and Nishihara, Robert and Jordan, Michael},
          booktitle = 	 {Proceedings of the 19th International Conference on Artificial Intelligence and Statistics},
          pages = 	 {249--258},
          year = 	 {2016},
          editor = 	 {Gretton, Arthur and Robert, Christian C.},
          volume = 	 {51},
          series = 	 {Proceedings of Machine Learning Research},
          address = 	 {Cadiz, Spain},
          month = 	 {09--11 May},
          publisher =    {PMLR},
          pdf = 	 {http://proceedings.mlr.press/v51/moritz16.pdf},
          url = 	 {https://proceedings.mlr.press/v51/moritz16.html},
          abstract = 	 {We propose a new stochastic L-BFGS algorithm and prove a linear convergence rate for strongly convex and smooth functions.
                        Our algorithm draws heavily from a recent stochastic variant of L-BFGS proposed in Byrd et al. (2014)
                        as well as a recent approach to variance reduction for stochastic gradient descent from Johnson and Zhang (2013).
                        We demonstrate experimentally that our algorithm performs well on large-scale convex and non-convex optimization problems,
                        exhibiting linear convergence and rapidly solving the optimization problems to high levels of precision.
                        Furthermore, we show that our algorithm performs well for a wide-range of step sizes, often differing by several orders of magnitude.}
        }"""
        raise NotImplementedError
