import numpy as np
from matplotlib import pyplot as plt

from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevRx
from squlearn.observables import IsingHamiltonian
from squlearn.optimizers import Adam
from squlearn.qnn import QNNRegressor, SquaredLoss

executor = Executor("qasm_simulator")
executor.set_shots(5000)

# define the PQC
nqubits = 10
number_of_layers = 3
pqc = ChebyshevRx(nqubits, 1, num_layers=number_of_layers)

# define the Observable
ising_op = IsingHamiltonian(nqubits, I="S", Z="S", ZZ="S")

# define random initial parameters for the PQC and the cost operator
np.random.seed(13)
param_ini = np.random.rand(pqc.num_parameters)
param_op_ini = np.random.rand(ising_op.num_parameters)

# define the optimzer options
x0 = [[i * 0.02] for i in range(15)]
optimizer_options = {"bo_aqc_func": "EI", "bo_aqc_optimizer": "lbfgs", "bo_bounds": [(0.0, 0.3)], "log_file": "/data/log_sglbo_noiseless",
                     "bo_n_calls": 30, "bo_x0_points": x0, "maxiter": 200}

# define the regressor
reg = QNNRegressor(pqc, ising_op, executor, SquaredLoss(), Adam(optimizer_options), param_ini, param_op_ini)

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
plt.plot(x, np.log(x), label="Tatsächliche Log-Funktion")
plt.plot(x, y, label="Vorhergesagte Funktion")

# plot the error of the QNN
plt.plot(x, np.abs(y - np.log(x)), label="Fehler")
plt.legend()

plt.tight_layout()
plt.savefig('/data/regression.pdf')