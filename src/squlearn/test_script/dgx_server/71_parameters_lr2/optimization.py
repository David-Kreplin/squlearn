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

qnn_simulator_adam = QNNRegressor(
        pqc,
        op,
        Executor("statevector_simulator"),
        SquaredLoss(),
        Adam({"lr": 0.3, "log_file": f"/data/adam_{num_parameters}_params_new.log", "maxiter": 500}),
        param_ini,
        param_op_ini=param_op_ini,
        opt_param_op=True,  
        parameter_seed=124
    )

# Data that is inputted to the QNN
x_train = np.arange(-0.5, 0.6, 0.1)
# Data that is fitted by the QNN
y_train = np.sin(6.0 * x_train)

qnn_simulator_adam.fit(x_train, y_train)
