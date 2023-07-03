from squlearn.feature_map.qcnn_feature_map import qcnn_feature_map
import dill as pickle
from squlearn.util import Executor
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
import itertools
import random
from squlearn.expectation_operator import SingleAmplitude
from squlearn.qnn import QNN
from squlearn.feature_map.layered_feature_map import LayeredFeatureMap
from qiskit.primitives import Estimator
from squlearn.qnn.training import regression
from squlearn.optimizers import SLSQP,Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.qnn.qnnr import QNNRegressor

def generate_data_all_combi(length):
    """Creates a 2 dimensional numpy array with all combinations of zeros and ones.
      It outputs this with there labels (0 if more zeros than ones, 1 else)"""
    all_combi_list = list(itertools.product([0,1],repeat=length))
    X_tuple_list = all_combi_list
    X = np.array(X_tuple_list)  #converts list of tuples into a numpy array with dimension 2
    Y = np.count_nonzero(X==0,axis=1) #counts the number of zeros in each sublist
    for i in range(len(Y)):
        zeros = Y[i]
        if 2*zeros > length:
            Y[i] = 0
        else:
            Y[i] = 1 
    return X,Y

def generate_train_data(all_combination_data,train_size):
    """Generates from given all_combination data a train set with there labels."""
    X,Y = all_combination_data[0],all_combination_data[1]
    data_size = Y.size
    index_list = range(data_size)
    random.seed(0)
    index_choice = random.sample(index_list, train_size)
    X_train = np.array([X[i] for i in index_choice])
    Y_train = np.array([Y[i] for i in index_choice])
    return X_train,Y_train

qubits = 6
train_set_size = 32
#-------------------------------------------------------------------------------------------------------------------------------------------
# QCNN feature map for training:
"""
We apply the following gates:
Rx: for encoding
U_0: convolution gate (crx entangled with 3 qubits)
V_1: convolution gate (crz entangled with 2 qubits)
P_0: pooling gate (cx with 2 qubits, controlled qubit: 0, target qubit: 1)
U_2: U_0 again
V_3: V_1 again
P_1: P_0 again
F: fully connected layer (crx entangled with 2 qubits)
"""
qcnn_train = qcnn_feature_map(qubits)
x = ParameterVector("x", 3)
gate_qc = QuantumCircuit(3)
gate_qc.crx(x[0],0,1)
gate_qc.crx(x[2],0,2)
gate_qc.crx(x[1],1,2)
y = ParameterVector("y", 2)
gate_conv2 = QuantumCircuit(2)
gate_conv2.crz(y[0],0,1)
gate_conv2.crz(y[1],1,0)
pooling_gate1 = QuantumCircuit(2)
pooling_gate1.cx(0,1)
# Fully for 2 qubits:
a = ParameterVector("a",2)
fully_gate = QuantumCircuit(2)
fully_gate.crx(a[0],0,1)
fully_gate.crx(a[1],1,0)
qcnn_train.convolution(gate_qc, entangled=False,operator="U",var_param = False)
qcnn_train.convolution(gate_conv2, entangled=False,operator="V",var_param = False)
qcnn_train.pooling(pooling_gate1,operator="P")
qcnn_train.convolution(gate_qc, entangled=False,operator="U",var_param = False)
qcnn_train.convolution(gate_conv2, entangled=False,operator="V",var_param = False)
qcnn_train.pooling(pooling_gate1,operator="P")
qcnn_train.fully_connected(fully_gate)
param_vec_for_qcnn = ParameterVector("p", qcnn_train.num_parameters)
qcnn_train.get_circuit([],param_vec_for_qcnn)
#-------------------------------------------------------------------------------------------------------------------------------------------

# Measure |0> state in qubit 3
operator = SingleAmplitude(qubits,3)

all_combination_data = generate_data_all_combi(qubits)
X,Y = generate_train_data(all_combination_data,train_set_size)

encode = LayeredFeatureMap.from_string("Rx(x)",num_qubits=qubits,num_features=qubits)
qcnn_pqc = encode + qcnn_train
qcnn_pqc.draw()

np.random.seed(13) 
param_ini = np.random.rand(qcnn_pqc.num_parameters)
param_op_ini = np.random.rand(operator.num_parameters)
reg = QNNRegressor(qcnn_pqc,
                   operator,
                   Executor(Estimator()),
                   SquaredLoss,
                   Adam({"maxiter":150}),
                   param_ini,
                   param_op_ini,
                   opt_param_op=False,
                   batch_size=5,
                   epochs=50,
                   shuffle=True,
                   )

reg.fit(X, Y)

print("Test with",qubits,"qubits and",train_set_size,"train sets.")
print("Train data:")
print((X,Y))
X_test = all_combination_data[0]
Y_test = all_combination_data[1]
print("Test data:")
print(X_test)
print("predict",reg.predict(X_test))
print("ref",Y_test)
print("Score:",reg.score(X_test,Y_test))


pickle.dump( reg , open( "/data/model_train_qcnn_with_32_data_points.p", "wb" ) )



