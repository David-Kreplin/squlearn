import abc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import rosen
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Real
from squlearn import Executor
from squlearn.encoding_circuit import ChebPQC, ChebRx
from squlearn.observables import SummedPaulis, IsingHamiltonian
from squlearn.optimizers import FiniteDiffGradient, SLSQP
from squlearn.optimizers.optimizer_base import OptimizerBase, SGDMixin, default_callback, OptimizerResult
from squlearn.qnn import QNNRegressor, SquaredLoss, QNNClassifier


class SGLBO(OptimizerBase, SGDMixin):
    """sQUlearn's implementation of the SGLBO optimizer
    Possible options that can be set in the options dictionary are:

    * **tol** (float): Tolerance for the termination of the optimization (default: 1e-6)
    * **maxiter** (int): Maximum number of iterations per fit run (default: 100)
    * **maxiter_total** (int): Maximum number of iterations in total (default: maxiter)
    * **eps** (float): Step size for finite differences (default: 0.01)
    * **bo_calls** (int): Number of iterations for the Bayesian Optimization (default: 10)
    * **bo_bounds** (List): Lower and upper bound for the search space for the Bayesian Optimization for each dimension. Each bound should be provided in the `skopt.space.Real` format (default: (0.001, 0.01))

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
        self.bo_calls = options.get("bo_calls", 10)
        self.bo_bounds = options.get("bo_bounds", [Real(0.001, 0.01)])

        self.callback = callback
        self.options = options
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

        optimal_step_size = self.__optimal_step_size(self.func, self.x, grad)
        update = optimal_step_size * grad

        print("optimal step Size:", optimal_step_size)
        print("update:", update)
        print("gradient:", grad)

        return -update

    def __optimal_step_size(self, func, start_point, gradient):
        # cost function to optimize the step size in one dimension
        def step_size_cost(x):
            updated_point = start_point.copy()
            updated_point = updated_point - x * gradient
            return func(updated_point)

        # bayesian optimization to estimate the step size in one dimension
        result = gp_minimize(step_size_cost, self.bo_bounds, n_calls=self.bo_calls)

        return result.x

    def _update_lr(self) -> None:
        pass


def regression_example_logarithm(optimizer):
    """
    In this example a QNNRegressor is trained to fit a logarithm

    Args:
        optimizer (OptimizerBase): The Optimizer used to optimize the loss function
    """
    executor = Executor("statevector_simulator")

    # define the PQC
    nqubits = 4
    number_of_layers = 2
    pqc = ChebRx(nqubits, 1, num_layers=number_of_layers)

    # define the Observable
    ising_op = IsingHamiltonian(nqubits, I="S", Z="S", ZZ="S")

    # define random initial parameters for the PQC and the cost operator
    np.random.seed(13)
    param_ini = np.random.rand(pqc.num_parameters)
    param_op_ini = np.random.rand(ising_op.num_parameters)

    # define the regressor
    reg = QNNRegressor(pqc, ising_op, executor, SquaredLoss(), optimizer, param_ini, param_op_ini)

    # train the regressor
    x_space = np.arange(0.1, 0.9, 0.1)
    ref_values = np.log(x_space)

    reg.fit(x_space, ref_values)

    # print the trained parameters for the PQC and the operator
    print("Result PQC params:", reg.param)
    print("Result operator params:", reg.param_op)

    # plot the predicted function vs. the actual logarithm function
    x = np.arange(np.min(x_space), np.max(x_space), 0.005)
    y = reg.predict(x)
    plt.plot(x, np.log(x))
    plt.plot(x, y)

    # plot the error of the QNN
    plt.plot(x, np.abs(y - np.log(x)))


def classification_example(optimizer):
    """
    In this example a QNNClassifier is used for classification of points in 2D space

    Args:
        optimizer (OptimizerBase): Optimizer used to optimize the loss function
    """
    executor = Executor("statevector_simulator")

    # define the PQC
    nqubits = 4
    number_of_layers = 2
    pqc = ChebRx(nqubits, 2, num_layers=number_of_layers)

    # define the observable
    cost_op = SummedPaulis(nqubits)

    # define initial parameters for the PQC and the cost operator
    np.random.seed(24)
    param_ini = np.random.rand(pqc.num_parameters)
    param_op_ini = np.random.rand(cost_op.num_parameters)

    # define the classifier
    clf = QNNClassifier(pqc, cost_op, executor, SquaredLoss(), optimizer, param_ini, param_op_ini)

    # generate the dataset
    X, y = make_blobs(60, centers=2, random_state=0)
    X = MinMaxScaler((-0.9, 0.9)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train the classifier
    clf.fit(X_train, y_train)

    # print the trained parameters
    print("Result PQC params:", clf.param)
    print("Result operator params:", clf.param_op)

    # check the classifiers performance on the test set
    clf.score(X_test, y_test)

    # visualize the training and test data, as well as the decision boundary of the classifier
    xx, yy = np.meshgrid(np.arange(-0.99, 0.99, 0.01), np.arange(-0.99, 0.99, 0.01))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.6)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, marker="X")
    plt.plot()


if __name__ == '__main__':

    regression_example_logarithm(SGLBO())
