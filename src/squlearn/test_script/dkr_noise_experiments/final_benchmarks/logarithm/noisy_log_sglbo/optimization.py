import numpy as np
import matplotlib.pyplot as plt

from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevRx
from squlearn.observables import SummedPaulis
from squlearn.qnn import QNNRegressor, SquaredLoss
from squlearn.optimizers import Adam, SGLBO

executor = Executor("qasm_simulator")
executor.set_shots(5000)

nqubits = 4
number_of_layers = 2

pqc = ChebyshevRx(nqubits, 1, num_layers=number_of_layers)

op = SummedPaulis(nqubits)
print(op)


# Randomly initialize parameters of the encoding circuit
np.random.seed(13)
param_ini =  np.random.rand(pqc.num_parameters)
# Initialize parameters of the observable as ones
param_op_ini = np.random.rand(op.num_parameters)

#x0 = [[i * 0.02] for i in range(15)]
x0 = [[1e-6], [2e-6], [1e-5], [2e-5]]
optimizer_options = {"tol":0.0, "bo_aqc_func": "EI", "bo_aqc_optimizer": "lbfgs", "bo_bounds": [(-1e-3, 0.2)], "log_file": "/data/sglbo_log_noise.log",
                     "bo_n_calls": 60, "bo_x0_points": x0, "maxiter": 300, }

qnn_simulator_sglbo = QNNRegressor(
    pqc,
    op,
    executor,
    SquaredLoss(),
    SGLBO(optimizer_options),
    param_ini,
    param_op_ini=param_op_ini,
)

x_train = np.arange(0.1, 0.9, 0.1)
y_train = np.log(x_train)

qnn_simulator_sglbo.fit(x_train, y_train)