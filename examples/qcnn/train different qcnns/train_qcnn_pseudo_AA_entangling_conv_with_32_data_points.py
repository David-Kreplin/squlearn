from squlearn.feature_map.qcnn_feature_map import qcnn_feature_map
from squlearn.feature_map.qcnn_feature_map import prototype_routine
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
    index_choice = random.sample(index_list, train_size)
    X_train = np.array([X[i] for i in index_choice])
    Y_train = np.array([Y[i] for i in index_choice])
    return X_train,Y_train

qubits = 6
train_set_size = 32
#-------------------------------------------------------------------------------------------------------------------------------------------
#QCNN feature map with one 5 gate convolution layer with a pseudo all in all entangling
qcnn_train = qcnn_feature_map(qubits)

x = ParameterVector("x",11)
conv_gate1 = QuantumCircuit(5)
conv_gate1.crx(x[0],0,1)
conv_gate1.crx(x[1],0,2)
conv_gate1.crx(x[2],0,3)
conv_gate1.crx(x[3],0,4)
conv_gate1.crx(x[4],1,2)
conv_gate1.crx(x[5],1,3)
conv_gate1.crx(x[6],1,4)
conv_gate1.crx(x[7],2,3)
conv_gate1.crx(x[8],2,4)
conv_gate1.crx(x[9],3,4)
conv_gate1.crx(x[10],4,0)

pool_gate = QuantumCircuit(2)
pool_gate.cx(0,1)

z = ParameterVector("z",7)
conv_gate2 = QuantumCircuit(3)
conv_gate2.crz(z[0],0,1)
conv_gate2.crz(z[1],0,2)
conv_gate2.crz(z[2],1,2)
conv_gate2.crz(z[3],2,0)

fully_gate = QuantumCircuit(2)
fully_gate.cry(z[4],0,1)
fully_gate.cry(z[5],1,0)

qcnn_train.convolution(conv_gate1,entangled=True,var_param=True)
qcnn_train.pooling(pool_gate)
qcnn_train.convolution(conv_gate2)
qcnn_train.pooling(pool_gate)
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
                   Adam({"maxiter":100}),
                   param_ini,
                   param_op_ini,
                   opt_param_op=False,
                   batch_size=10,
                   epochs=30,
                   shuffle=True,
                   )

reg.fit(X, Y)

print("Test with",qubits,"qubits and",train_set_size,"train sets.")
print("Train data:")
print((X,Y))
X_test = all_combination_data[0]
print("Test data:")
print(X_test)
print(reg.predict(X_test))
