"""Tests for training methods."""
import numpy as np

import pytest

from squlearn import Executor
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.qnn.qnn import QNN
from squlearn.qnn import QNNRegressor
from squlearn.qnn.training import train_mini_batch, ShotsFromRSTD

executor = Executor("statevector_simulator")

examples = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]


class TestSolvemini_batch:
    """Tests for mini-batch gradient descent."""

    pqc = ChebyshevPQC(4, 1, 3, False)
    cost_op = SummedPaulis(4)
    qnn = QNN(pqc, cost_op, executor)
    ex_1 = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]

    def test_wrong_optimizer(self):
        """Test for error caused by wrong optimizer type."""
        param_ini = np.random.rand(self.qnn.num_parameters) * 4
        param_op_ini = np.ones(self.qnn.num_qubits + 1)
        with pytest.raises(TypeError, match="is not supported for mini-batch gradient descent."):
            train_mini_batch(
                self.qnn,
                self.ex_1[0],
                self.ex_1[1],
                param_ini,
                param_op_ini,
                loss=SquaredLoss(),
                optimizer=SLSQP(),
                batch_size=10,
                epochs=30,
            )


class TestShotsFromRSTD:
    def test_qnn_training(self):
        """Test a optimization with variance reduction and shots from RSTD."""

        pqc = ChebyshevPQC(2, 1, 3, False)
        ob = SummedPaulis(2)
        executor = Executor("qasm_simulator", primitive_seed=0)
        qnn = QNNRegressor(
            pqc,
            ob,
            executor,
            SquaredLoss(),
            Adam(options={"lr": 0.3, "maxiter": 3}),
            variance=0.005,
            shot_control=ShotsFromRSTD(),
            parameter_seed=0,
        )
        x_train = np.arange(-0.2, 0.3, 0.1)
        y_train = np.abs(x_train)
        qnn.fit(x_train, y_train)
        test = qnn.predict(x_train)
        reference = np.array([0.31176001, 0.09348281, -0.05118243, -0.25693387, -0.43025503])
        assert np.allclose(test, reference)
