from squlearn.encoding_circuit import QiskitEncodingCircuit, ChebyshevPQC
from squlearn.observables import SummedPaulis
from squlearn.optimizers import Adam, SGLBO
from squlearn.qnn import QNNRegressor
from squlearn import Executor
from squlearn.qnn import SquaredLoss

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import matplotlib.pyplot as plt

pqc = ChebyshevPQC(10,1,2)

nqubits = 10
op = SummedPaulis(nqubits)

# Randomly initialize parameters of the encoding circuit
np.random.seed(13)
param_ini = np.random.rand(pqc.num_parameters)
# Initialize parameters of the observable as ones
param_op_ini = np.ones(op.num_parameters)

num_parameters = pqc.num_parameters + op.num_parameters

x0 = [[1e-6], [2e-6], [3e-6], [4e-6], [5e-6], [6e-6], [7e-6], [8e-6], [9e-6], [1e-5], [2e-5], [3e-5], [4e-5], [5e-5], [6e-5], [7e-5], [8e-5], [9e-5], [1e-4], [2e-4], [3e-4], [4e-4], [5e-4], [6e-4], [7e-4], [8e-4], [9e-4], [1e-3], [2e-3], [3e-3], [4e-3], [5e-3], [6e-3], [7e-3], [8e-3], [9e-3], [1e-2], [2e-2], [3e-2], [4e-2], [5e-2], [6e-2], [7e-2], [8e-2], [9e-2], [1e-1], [2e-1]]
optimizer_options = {"bo_aqc_func": "EI", "bo_aqc_optimizer": "lbfgs", "bo_bounds": [(0.0, 0.2)], "log_file": f"sglbo_{num_parameters}_params_new.log", "bo_n_calls": 60, "bo_x0_points": x0, "maxiter": 500}

qnn_simulator_sglbo = QNNRegressor(
        pqc,
        op,
        Executor("statevector_simulator"),
        SquaredLoss(),
        SGLBO(optimizer_options),
        param_ini,
        param_op_ini=param_op_ini,
        opt_param_op=True,  # Keine Observablen optimierung
        parameter_seed=124
    )

# Data that is inputted to the QNN
x_train = np.arange(-0.5, 0.6, 0.1)
# Data that is fitted by the QNN
y_train = np.sin(6.0 * x_train)

qnn_simulator_sglbo.fit(x_train, y_train)
