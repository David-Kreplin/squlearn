import abc
import numpy as np
from skopt import gp_minimize, expected_minimum

from .approximated_gradients import FiniteDiffGradient
from .optimizer_base import OptimizerBase, SGDMixin, default_callback, OptimizerResult

class SGLBO(OptimizerBase, SGDMixin):
    """sQUlearn's implementation of the SGLBO optimizer
    Possible options that can be set in the options dictionary are:

    * **tol** (float): Tolerance for the termination of the optimization (default: 1e-6)
    * **maxiter** (int): Maximum number of iterations per fit run (default: 100)
    * **maxiter_total** (int): Maximum number of iterations in total (default: maxiter)
    * **eps** (float): Step size for finite differences (default: 0.01)
    * **bo_n_calls** (int): Number of iterations for the Bayesian Optimization (default: 20)
    * **bo_bounds** (List): Lower and upper bound for the search space for the Bayesian Optimization for each dimension. Each bound should be provided as a tupel (default: (0.0, 0.3))
    * **bo_n_initial_points** (int): Number of initial points for the Bayesian Optimization (default: 10)
    * **bo_x0_points** (list of lists): Initial input points (default: None)
    * **bo_aqc_optimizer** (str): Method to minimize the acquisition function. "sampling" or "lbfgs" (default: "lbfgs")
    * **bo_acq_func** (str): Acquisition function for the Bayesian Optimization (default: "EI").
    * **bo_noise** (float): Noise for noisy observations (default: "gaussian")
      Valid values for `acq_func` are:
        * "LCB"
        * "EI"
        * "PI"
        * "gp_hedge"
    * **log_file** (str): File to log the optimization (default: None)

    Args:
        options (dict): Options for the SGLBO optimizer
    """

    def __init__(self, options: dict = None, callback=default_callback) -> None:
        super(SGDMixin, self).__init__()

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        self.maxiter = options.get("maxiter", 100)
        self.maxiter_total = options.get("maxiter_total", self.maxiter)
        self.eps = options.get("eps", 0.01)
        self.bo_n_calls = options.get("bo_n_calls", 20)
        self.bo_bounds = options.get("bo_bounds", [(0.0, 0.3)])
        self.bo_aqc_func = options.get("bo_aqc_func", "EI")
        self.bo_aqc_optimizer = options.get("bo_aqc_optimizer", "lbfgs")
        self.bo_n_initial_points = options.get("bo_n_initial_points", 10)
        self.bo_x0_points = options.get("bo_x0_points")
        self.bo_noise = options.get("bo_noise", "gaussian")
        self.log_file = options.get("log_file", None)

        self.callback = callback
        self.options = options
        self.x = None
        self.func = None

        if self.log_file is not None:
            f = open(self.log_file, "w")
            header = (f"maxiter_total: {self.maxiter_total}\n"
                      f"bo_n_calls: {self.bo_n_calls}\n"
                      f"bo_bounds: {self.bo_bounds}\n"
                      f"bo_aqc_func: {self.bo_aqc_func}\n"
                      f"bo_aqc_optimizer: {self.bo_aqc_optimizer}\n"
                      f"bo_n_initial_points: {self.bo_n_initial_points}\n"
                      f"bo_x0_points: {self.bo_x0_points}\n")
            output = " %9s  %12s  %12s  %12s \n" % (
                "Iteration",
                "f(x)",
                "Gradient",
                "Step"
            )
            f.write(header)
            f.write(output)
            f.close()

    def minimize(
            self,
            fun: callable,
            x0: np.ndarray,
            grad: callable = None,
            bounds=None,
    ) -> OptimizerResult:
        """
        Function to minimize a given function using the SGLBO optimizer.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """

        self.func = fun

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
            fval = fun(self.x)
            gradient = grad(self.x)

            x_updated = self.step(x=self.x, grad=gradient)

            # check termination
            if np.linalg.norm(self.x - x_updated) < self.tol:
                break

            if self.log_file is not None:
                self._log(fval, gradient, np.linalg.norm(self.x - x_updated))

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

        optimal_step_size = self.__optimal_step_size(self.func, self.x, grad)
        update = optimal_step_size * grad
        return -update

    def __optimal_step_size(self, func, start_point, gradient):
        # cost function to optimize the step size in one dimension
        def step_size_cost(x):
            updated_point = start_point.copy()
            updated_point = updated_point - x * gradient
            print("BOP ", "fval: ", func(updated_point), " x: ", x)
            return func(updated_point)

        # bayesian optimization to estimate the step size in one dimension
        res = gp_minimize(step_size_cost, self.bo_bounds, n_calls=self.bo_n_calls, acq_func=self.bo_aqc_func,
                             acq_optimizer=self.bo_aqc_optimizer, x0=self.bo_x0_points, n_jobs=-1, random_state=0, noise=self.bo_noise)
        fun, x = expected_minimum(res)
        print('\033[91m', "Iteration: ", self.iteration, ": ", "gp_minimize: ", "fval: ", fun, " x: ", x, '\033[0m')

        return x

    def _update_lr(self) -> None:
        pass

    def _log(self, fval, gradient, dx):
        """Function for creating a log entry of the optimization."""
        if self.log_file is not None:
            f = open(self.log_file, "a")
            if fval is not None:
                output = " %9d  %12.5f  %12.5f  %12.5f  \n" % (
                    self.iteration,
                    fval,
                    np.linalg.norm(gradient),
                    dx,
                )
            else:
                output = " %9d  %12.5f  %12.5f  \n" % (
                    self.iteration,
                    np.linalg.norm(gradient),
                    dx,
                )
            f.write(output)
            f.close()
