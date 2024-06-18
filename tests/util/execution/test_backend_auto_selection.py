import pytest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeManila, FakeBelem, FakeAthens
from qiskit_aer import Aer
from squlearn.util import Executor
from squlearn.qnn import QNNRegressor
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.optimizers import Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, QKRR


class TestBackendAutoSelection:

    def test_auto_select_circuit(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        backends = [FakeBelem(), FakeAthens(), FakeManila()]
        executor = Executor(backends, seed=0, shots=10000)
        qc2, info = executor.select_backend(qc)
        assert str(executor.backend) == "fake_manila"
        sampler = executor.get_sampler()
        result = sampler.run(qc.measure_all(inplace=False)).result()
        assert result.metadata[0]["shots"] == 10000
        assert result.quasi_dists[0] == {2: 0.229, 3: 0.7378, 0: 0.0071, 1: 0.0261}
        estimator = executor.get_estimator()
        result = estimator.run(qc, obs).result()
        assert result.values[0] == 0.5008
        assert result.metadata[0]["shots"] == 10000

    def test_auto_select_circuit_parallel(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        backends = [FakeBelem(), FakeAthens(), FakeManila()]
        executor = Executor(backends, seed=0, shots=10000, qpu_parallelization=2)
        qc2, info = executor.select_backend(qc)
        assert str(executor.backend) == "fake_belem"
        sampler = executor.get_sampler()
        result = sampler.run(qc.measure_all(inplace=False)).result()
        assert result.metadata[0]["shots"] == 10000
        assert round(result.quasi_dists[0][0], 1) == 0.0
        assert round(result.quasi_dists[0][1], 1) == 0.1
        assert round(result.quasi_dists[0][2], 1) == 0.0
        assert round(result.quasi_dists[0][3], 1) == 0.9
        estimator = executor.get_estimator()
        result = estimator.run(qc, obs).result()
        assert round(result.values[0], 1) == 0.8
        assert result.metadata[0]["shots"] == 10000

    def test_auto_select_qnn(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelem(), FakeAthens(), FakeManila()]
        executor = Executor(backends, seed=0, shots=10000)
        pqc = ChebyshevTower(2, 1, 2)
        obs = SummedPaulis(2)
        param = []
        param_op = np.ones(obs.num_parameters)
        qnn = QNNRegressor(
            pqc,
            obs,
            executor,
            SquaredLoss(),
            Adam(),
            pretrained=True,
            param_ini=param,
            param_op_ini=param_op,
        )
        value = qnn.predict(np.array([0.25, 0.75]))
        assert str(executor.backend) == "fake_manila"
        assert np.allclose(value, np.array([-0.3322, -0.5342]), atol=1e-3)

    def test_auto_select_fidelity_kernel(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelem(), FakeAthens(), FakeManila()]
        executor = Executor(backends, seed=0, shots=10000)
        pqc = ChebyshevTower(2, 1, 2)
        fqk = FidelityKernel(pqc, executor)
        qkrr = QKRR(fqk)
        qkrr.fit(np.array([[0.25], [0.75]]), np.array([0.25, 0.75]))
        value = qkrr.predict(np.array([[0.25], [0.75]]))
        assert str(executor.backend) == "fake_manila"
        assert np.allclose(value, np.array([0.25828537766263315, 0.7475716159340855]), atol=1e-3)

    def test_auto_select_projected_kernel(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelem(), FakeAthens(), FakeManila()]
        executor = Executor(backends, seed=0, shots=10000)
        pqc = ChebyshevTower(2, 1, 2)
        fqk = ProjectedQuantumKernel(pqc, executor)
        qkrr = QKRR(fqk)
        qkrr.fit(np.array([[0.25], [0.75]]), np.array([0.25, 0.75]))
        value = qkrr.predict(np.array([[0.25], [0.75]]))
        assert str(executor.backend) == "fake_manila"
        assert np.allclose(value, np.array([0.24999992453371855, 0.7499992679943152]), atol=1e-3)
