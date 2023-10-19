import abc
import numpy as np
from scipy.optimize import rosen
from skopt import gp_minimize
from skopt.space import Real

from src.squlearn.optimizers import FiniteDiffGradient
from src.squlearn.optimizers.optimizer_base import OptimizerBase, SGDMixin, OptimizerResult


class SGLBO(OptimizerBase, SGDMixin):
    """sQUlearn's implementation of the SGLBO optimizer
    Possible options that can be set in the options dictionary are:

    * **tol** (float): Tolerance for the termination of the optimization (default: 1e-6)
    * **maxiter** (int): Maximum number of iterations per fit run (default: 100)
    * **maxiter_total** (int): Maximum number of iterations in total (default: maxiter)
    * **eps** (float): Step size for finite differences (default: 0.01)
    * **bo_calls** (int): Number of iterations for the Bayesian Optimization (default: 10)
    * **bo_bounds** (List): Lower and upper bound for the search space for the Bayesian Optimization for each dimension (default: (0.001, 0.01))

    Args:
        options (dict): Options for the SGLBO optimizer
    """

    def __init__(self, options: dict = None) -> None:
        super(SGDMixin, self).__init__()

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        self.maxiter = options.get("maxiter", 100)
        self.maxiter_total = options.get("maxiter_total", self.maxiter)
        self.eps = options.get("eps", 0.01)
        self.bo_calls = options.get("bo_calls", 10)
        self.bo_bounds = options.get("bo_bounds", [])

        self.x = None
        self.func = None

    def minimize(
            self,
            fun: callable,
            x0: np.ndarray,
            grad: callable = None,
            bounds=None,
    ) -> OptimizerResult:
        """
        Function to minimize a given function using the ADAM optimizer.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """

        self.func = fun
        self.bo_bounds = [Real(0.01, 0.1) for _ in range(len(x0))]

        # set-up number of iterations of the current run (needed for restarts)
        if self.maxiter != self.maxiter_total:
            maxiter = self.iteration + self.maxiter
        else:
            maxiter = self.maxiter_total

        self.x = x0

        if grad is None:
            grad = FiniteDiffGradient(fun, eps=self.eps).gradient

        while self.iteration < maxiter:
            # calculate the gradient
            gradient = grad(self.x)

            x_updated = self.step(x=self.x, grad=gradient)

            # check termination
            if np.linalg.norm(self.x - x_updated) < self.tol:
                break

            self.x = x_updated

        result = OptimizerResult()
        result.x = self.x
        result.fun = fun(self.x)
        result.nit = self.iteration

        return result

    def _get_update(self, grad: np.ndarray) -> np.ndarray:
        """Function that returns the update for a given gradient.

        Args:
            grad (np.ndarray): Gradient of the objective function.

        Returns:
            Update for the current iteration (np.ndarray).

        """
        update = np.zeros_like(self.x)

        for i in range(len(self.x)):
            gradient = grad[i]
            optimal_step_size = self.__optimal_step_size(self.func, self.x, gradient, i)
            update[i] = optimal_step_size * gradient

        return update

    def __optimal_step_size(self, func, start_point, gradient, dimension):
        # cost function to optimize the step size in one dimension
        def step_size_cost(x):
            updated_point = start_point.copy()
            updated_point[dimension] = updated_point[dimension] - x[dimension] * gradient
            return func(updated_point)

        # bayesian optimization to estimate the step size in one dimension
        result = gp_minimize(step_size_cost, self.bo_bounds, n_calls=self.bo_calls)

        return result.x[dimension]

    def _update_lr(self) -> None:
        pass


def f(x):
    return x[0] ** 2


if __name__ == "__main__":
    start_point = np.array([5.0])

    sglbo = SGLBO()
    result = sglbo.minimize(f, start_point)

    print("fun:", result.fun, "x:", result.x)
