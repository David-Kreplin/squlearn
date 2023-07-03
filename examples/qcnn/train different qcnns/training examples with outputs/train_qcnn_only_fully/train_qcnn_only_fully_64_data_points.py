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
train_set_size = 64
#-------------------------------------------------------------------------------------------------------------------------------------------
# QCNN only with fully connected layer (so it's a qnn):
qcnn_train = qcnn_feature_map(6)
x = ParameterVector("x",18)
fully_gate = QuantumCircuit(6)
fully_gate.crx(x[0],0,1)
fully_gate.crx(x[1],1,2)
fully_gate.crx(x[2],2,3)
fully_gate.crx(x[3],3,4)
fully_gate.crx(x[4],4,5)
fully_gate.crx(x[5],5,0)

fully_gate.ry(x[6],0)
fully_gate.ry(x[7],1)
fully_gate.ry(x[8],2)
fully_gate.ry(x[9],3)
fully_gate.ry(x[10],4)
fully_gate.ry(x[11],5)

fully_gate.crz(x[12],0,1)
fully_gate.crz(x[13],1,2)
fully_gate.crz(x[14],2,3)
fully_gate.crz(x[15],3,4)
fully_gate.crz(x[16],4,5)
fully_gate.crz(x[17],5,0)

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


pickle.dump( reg , open( "/data/model_train_qcnn_only_fully_64_data_points.p", "wb" ) )