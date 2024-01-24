import abc
from typing import Union
import numpy as np

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor

from .lowlevel_qnn_base import LowLevelQNNBase


class LowLevelQNNPennyLane(LowLevelQNNBase):

    def __init__(self,
                 pqc: EncodingCircuitBase,
                 observable: Union[ObservableBase, list],
                 executor: Executor,
                 ) -> None:

        super().__init__(pqc, observable, executor)


        #self._pqc = pqc
        #self._observable = observable
        #self._executor = executor

    @abc.abstractmethod
    def set_params(self, **params) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self) -> dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        raise NotImplementedError

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        raise NotImplementedError

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        raise NotImplementedError

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        values,  # TODO: data type definition missing Union[str,Expec,tuple,...]
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:
        raise NotImplementedError

