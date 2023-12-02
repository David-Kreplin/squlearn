import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from squlearn import Executor
from squlearn.encoding_circuit import QiskitEncodingCircuit
from squlearn.observables import SinglePauli
from squlearn.qnn import QNNRegressor, SquaredLoss
from squlearn.optimizers import Adam, SLSQP, SGLBO



def pqc_1_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p2[1], 1)
    qc.ry(p2[2], 2)
    qc.ry(p2[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p2[4], 0)
    qc.ry(p2[5], 1)
    qc.ry(p2[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_2_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p2[2], 2)
    qc.ry(p2[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p2[4], 0)
    qc.ry(p2[5], 1)
    qc.ry(p2[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_3_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p[2], 2)
    qc.ry(p2[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p2[4], 0)
    qc.ry(p2[5], 1)
    qc.ry(p2[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_4_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p[2], 2)
    qc.ry(p[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p2[4], 0)
    qc.ry(p2[5], 1)
    qc.ry(p2[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_5_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p[2], 2)
    qc.ry(p[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[4], 0)
    qc.ry(p2[5], 1)
    qc.ry(p2[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_6_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p[2], 2)
    qc.ry(p[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[4], 0)
    qc.ry(p[5], 1)
    qc.ry(p2[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_7_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p[2], 2)
    qc.ry(p[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[4], 0)
    qc.ry(p[5], 1)
    qc.ry(p[6], 2)
    qc.ry(p2[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)
def pqc_8_parameter():

    qc = QuantumCircuit(4)
    p = ParameterVector('p', 8)
    x = ParameterVector('x', 1)

    p2 = np.array([-0.13005136, -1.34331017, 2.44392299, 1.64405423, -0.36315523, 0.45344925,
                   0.18577077, -0.13904446])

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.ry(p[2], 2)
    qc.ry(p[3], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    qc.rx(1 * np.arccos(x[0]), 0)
    qc.rx(2 * np.arccos(x[0]), 1)
    qc.rx(3 * np.arccos(x[0]), 2)
    qc.rx(4 * np.arccos(x[0]), 3)

    qc.ry(p[4], 0)
    qc.ry(p[5], 1)
    qc.ry(p[6], 2)
    qc.ry(p[7], 3)

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 0)

    return QiskitEncodingCircuit(qc)


for num_parameters in range(8):
    if num_parameters == 0:
        pqc = pqc_1_parameter()
    elif num_parameters == 1:
        pqc = pqc_2_parameter()
    elif num_parameters == 2:
        pqc = pqc_3_parameter()
    elif num_parameters == 3:
        pqc = pqc_4_parameter()
    elif num_parameters == 4:
        pqc = pqc_5_parameter()
    elif num_parameters == 5:
        pqc = pqc_6_parameter()
    elif num_parameters == 6:
        pqc = pqc_7_parameter()
    elif num_parameters == 7:
        pqc = pqc_8_parameter()

    nqubits = 4
    op = SinglePauli(nqubits, qubit=0, parameterized=True)

    # Randomly initialize parameters of the encoding circuit
    np.random.seed(13)
    param_ini = np.random.rand(pqc.num_parameters)
    # Initialize parameters of the observable as ones
    param_op_ini = np.ones(op.num_parameters)


    qnn_simulator = QNNRegressor(
        pqc,
        op,
        Executor("statevector_simulator"),
        SquaredLoss(),
        Adam({"lr": 0.2, "log_file": f"adam_params_{num_parameters + 1}.log"}),
        param_ini,
        param_op_ini=param_op_ini,
        opt_param_op=True,  # Keine Observablen optimierung
        parameter_seed=124
    )

    # Data that is inputted to the QNN
    x_train = np.arange(-0.5, 0.6, 0.1)
    # Data that is fitted by the QNN
    y_train = np.sin(6.0 * x_train)

    qnn_simulator.fit(x_train, y_train)
