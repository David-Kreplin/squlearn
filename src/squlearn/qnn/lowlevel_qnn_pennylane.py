import abc
from typing import Union
import numpy as np

from qiskit.circuit import ParameterVector

import pennylane as qml
import pennylane.numpy as pnp

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor

from ..util.pennylane import PennyLaneCircuit, PennyLaneDevice

from .lowlevel_qnn_base import LowLevelQNNBase

from ..util.data_preprocessing import adjust_features


class LowLevelQNNPennyLane(LowLevelQNNBase):

    def __init__(self,
                 pqc: EncodingCircuitBase,
                 observable: Union[ObservableBase, list],
                 executor: Executor,
                 result_caching: bool = False,
                 ) -> None:

        super().__init__(pqc, observable, executor)
        #self._pqc = pqc
        #self._observable = observable
        #self._executor = executor

        self._device = PennyLaneDevice()

        self._x = ParameterVector("x", self._pqc.num_features)
        self._param = ParameterVector("param", self._pqc.num_parameters)
        self._param_obs = ParameterVector("param_obs", self._observable.num_parameters)

        self._qiskit_circuit = self._pqc.get_circuit(self._x, self._param)
        self._qiskit_observable = self._observable.get_operator(self._param_obs)

        self._pennylane_circuit = PennyLaneCircuit(self._device, self._qiskit_circuit, self._qiskit_observable)

    def draw(self,**kwargs):

        return self._pennylane_circuit.draw(**kwargs)


    #@abc.abstractmethod
    def set_params(self, **params) -> None:
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> dict:
        """Returns the dictionary of the hyper-parameters of the QNN.

        In case of multiple outputs, the hyper-parameters of the operator are prefixed
        with ``op0__``, ``op1__``, etc.

        """
        params = dict(num_qubits=self.num_qubits)

        if deep:
            params.update(self._pqc.get_params())
            if isinstance(self._observable, list):
                for i, oper in enumerate(self._observable):
                    oper_dict = oper.get_params()
                    for key, value in oper_dict.items():
                        if key != "num_qubits":
                            params["op" + str(i) + "__" + key] = value
            else:
                params.update(self._observable.get_params())
        return params

    @property
    #@abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        return self._pennylane_circuit._num_qubits

    @property
    #@abc.abstractmethod
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        return self._pqc.num_features

    @property
    #@abc.abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        return self._pqc.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        return 1 #TODO

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self._observable.num_parameters

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return False # TODO

    #@abc.abstractmethod
    def evaluate(
        self,
        values,  # TODO: data type definition missing Union[str,Expec,tuple,...]
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:


        #x,test = adjust_features(x, self._pqc.num_features)


        if not isinstance(values, tuple):
            values = (values,)

        if self._pennylane_circuit.circuit_arguments != ["param","x","param_obs"]:
            raise NotImplementedError("Wrong order of circuit arguments!")

        value_dict = {}
        value_dict["x"] = x
        value_dict["param"] = param
        value_dict["param_op"] = param_obs
        xx = [x] # TODO: multiple features -> in a ([x1][x2]) list format (transposed to out format)


        for todo in values:
            if todo=="f" or values ==("f",):
                param_ = pnp.array(param, requires_grad=False)
                param_obs_ = pnp.array(param_obs, requires_grad=False)
                x_ = pnp.array(xx, requires_grad=False)
                value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
                value_dict["f"] = value
                # if "f" in value_dict:
                #     value_dict["f"].append(value)
                # else:
                #     value_dict["f"] = [value]
            elif todo=="dfdp" or values ==("dfdp",):
                param_ = pnp.array(param, requires_grad=True)
                param_obs_ = pnp.array(param_obs, requires_grad=False)
                x_ = pnp.array(xx, requires_grad=False)
                value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
                value_dict["dfdp"] = value
                # if "dfdp" in value_dict:
                #     value_dict["dfdp"].append(value)
                # else:
                #     value_dict["dfdp"] = [value]
            elif todo=="dfdop" or values ==("dfdop",):
                param_ = pnp.array(param, requires_grad=False)
                param_obs_ = pnp.array(param_obs, requires_grad=True)
                x_ = pnp.array(xx, requires_grad=False)
                value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
                value_dict["dfdop"] = value
                # if "dfdop" in value_dict:
                #     value_dict["dfdop"].append(value)
                # else:
                #     value_dict["dfdop"] = [value]
            elif todo=="dfdx" or values ==("dfdx",):
                param_ = pnp.array(param, requires_grad=False)
                param_obs_ = pnp.array(param_obs, requires_grad=False)
                x_ = pnp.array(xx, requires_grad=True)
                value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
                value_dict["dfdx"] = value
                # if "dfdx" in value_dict:
                #     value_dict["dfdx"].append(value)
                # else:
                #     value_dict["dfdx"] = [value]


        if "f" in value_dict:
            value_dict["f"] = np.array(value_dict["f"])
        if "dfdp" in value_dict:
            value_dict["dfdp"] = np.array(value_dict["dfdp"])
        if "dfdop" in value_dict:
            value_dict["dfdop"] = np.array(value_dict["dfdop"])
        if "dfdx" in value_dict:
            value_dict["dfdx"] = np.array(value_dict["dfdx"])

        return value_dict


    def evaluate_f(self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:

        return self.evaluate("f",x, param, param_obs)["f"]