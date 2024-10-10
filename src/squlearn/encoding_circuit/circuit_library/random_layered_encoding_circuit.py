import random
from typing import Union
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit


from ..layered_encoding_circuit import LayeredEncodingCircuit
from ..encoding_circuit_base import EncodingCircuitBase


class RandomLayeredEncodingCircuit(EncodingCircuitBase):
    r"""Creates a random Layered encoding circuit with no trainable parameters.

    A random Layered encoding circuit is generated by randomly selecting gates from a predefined
    action space. The action space contains single qubit gates, two qubit gates.
    Non-parametrized gates that are placed in the random circuit are: H, X, Y, Z, cx, cy, cz.
    Included paramererized gates are Rx, Ry, Rz, crx, cry, crz, where the parameter is either a
    fixed angle or a feature. Fixed angles are :math:`\pi`, :math:`\pi/2`, :math:`\pi/3`,
    :math:`\pi/4`, :math:`\pi/8`.
    Feature :math:`x` are encoded as :math:`x`, :math:`x\pi`, :math:`\arctan(x)`.
    The random circuit generation enforces, that all features are encoded at least once.

    A seed is used to identify the random circuit, so that the same circuit can be reproduced.
    The number of layers is randomly chosen between ``min_num_layers`` and ``max_num_layers``.
    The probability of a layer containing an encoding gate is given by ``feature_probability``.

    **Example for 4 qubits and a 6 dimensional feature vector**

    .. plot::

        from squlearn.encoding_circuit import RandomLayeredEncodingCircuit
        pqc = RandomLayeredEncodingCircuit(num_qubits=4, num_features=6)
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        num_features (int): Dimension of the feature vector
        seed (int): Seed for the random number generator (default: 0)
        min_num_layers (int): Minimum number of layers (default: 2)
        max_num_layers (int): Maximum number of layers (default: 10)
        feature_probability (float): Probability of a layer containing an encoding gate
                                     (default: 0.3)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        seed: int = 0,
        min_num_layers: int = 2,
        max_num_layers: int = 10,
        feature_probability=0.3,
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.seed = seed
        self.min_num_layers = min_num_layers
        self.max_num_layers = max_num_layers
        self.feature_probability = feature_probability
        self._fm_str = self._generate_circuit_string()
        self._layered_encoding_circuit = self._build_layered_encoding_circuit()

    def _generate_circuit_string(self) -> str:
        """Generates a random Layered encoding circuit string."""
        gates = [
            "H",
            "X",
            "Y",
            "Z",
            "cx",
            "cy",
            "cz",
            r"Rx(x)",
            r"Ry(x)",
            r"Rz(x)",
            r"Rx(x;=np.pi*(x),{x})",
            r"Ry(x;=np.pi*(x),{x})",
            r"Rz(x;=np.pi*(x),{x})",
            r"Rx(x;=np.arctan(x),{x})",
            r"Ry(x;=np.arctan(x),{x})",
            r"Rz(x;=np.arctan(x),{x})",
            r"Rx(p;=0*p+np.pi,{p})",
            r"Ry(p;=0*p+np.pi,{p})",
            r"Rz(p;=0*p+np.pi,{p})",
            r"crx(p;=0*p+np.pi,{p})",
            r"cry(p;=0*p+np.pi,{p})",
            r"crz(p;=0*p+np.pi,{p})",
            r"Rx(p;=0*p+1/2*np.pi,{p})",
            r"Ry(p;=0*p+1/2*np.pi,{p})",
            r"Rz(p;=0*p+1/2*np.pi,{p})",
            r"crx(p;=0*p+1/2*np.pi,{p})",
            r"cry(p;=0*p+1/2*np.pi,{p})",
            r"crz(p;=0*p+1/2*np.pi,{p})",
            r"Rx(p;=0*p+1/3*np.pi,{p})",
            r"Ry(p;=0*p+1/3*np.pi,{p})",
            r"Rz(p;=0*p+1/3*np.pi,{p})",
            r"crx(p;=0*p+1/3*np.pi,{p})",
            r"cry(p;=0*p+1/3*np.pi,{p})",
            r"crz(p;=0*p+1/3*np.pi,{p})",
            r"Rx(p;=0*p+1/4*np.pi,{p})",
            r"Ry(p;=0*p+1/4*np.pi,{p})",
            r"Rz(p;=0*p+1/4*np.pi,{p})",
            r"crx(p;=0*p+1/4*np.pi,{p})",
            r"cry(p;=0*p+1/4*np.pi,{p})",
            r"crz(p;=0*p+1/4*np.pi,{p})",
            r"Rx(p;=0*p+1/8*np.pi,{p})",
            r"Ry(p;=0*p+1/8*np.pi,{p})",
            r"Rz(p;=0*p+1/8*np.pi,{p})",
            r"crx(p;=0*p+1/8*np.pi,{p})",
            r"cry(p;=0*p+1/8*np.pi,{p})",
            r"crz(p;=0*p+1/8*np.pi,{p})",
        ]

        random.seed(self.seed)
        gates_with_x = [action for action in gates if "(x)" in action]
        weights = [
            self.feature_probability if "(x)" in action else 1 - self.feature_probability
            for action in gates
        ]
        num_layers = random.randint(self.min_num_layers, self.max_num_layers)
        layers = random.choices(gates, k=num_layers, weights=weights)

        min_x = (self.num_features - 1) // self.num_qubits + 1
        if min_x > self.min_num_layers:
            raise ValueError("Minimum number of layers is not enough to encode all features!")

        while str(layers).count("(x)") < min_x:
            layers.pop(-1)
            layers.append(random.choice(gates_with_x))
            random.shuffle(layers)

        return "-".join(layers)

    def _build_layered_encoding_circuit(self) -> LayeredEncodingCircuit:
        """Builds and returns the LayeredEncodingCircuit object from the fm_str."""
        return LayeredEncodingCircuit.from_string(
            encoding_circuit_str=self._fm_str,
            num_qubits=self.num_qubits,
            num_features=self.num_features,
        )

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = None,
    ) -> QuantumCircuit:
        r"""
        Returns the quantum circuit of the Random Layered encoding circuit.

        Args:
            features (Union[ParameterVector, np.ndarray]): The input features.
            parameters (Union[ParameterVector, np.ndarray]): The trainable parameters of the
                                                             circuit (not used, since there are
                                                             no free parameters).

        Returns:
            QuantumCircuit: The quantum circuit of the Random Layered encoding circuit.
        """

        parameter = np.zeros(self._layered_encoding_circuit.num_parameters)

        return self._layered_encoding_circuit.get_circuit(features, parameter)

    def get_params(self, deep: bool = True) -> dict:
        r"""
        Returns hyper-parameters and their values of the Random Layered encoding circuit

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["seed"] = self.seed
        params["min_num_layers"] = self.min_num_layers
        params["max_num_layers"] = self.max_num_layers
        params["feature_probability"] = self.feature_probability
        return params

    def set_params(self, **params) -> EncodingCircuitBase:
        r"""
        Sets value of the random layered encoding circuit hyper-parameters.

        The random circuit is regenerated with the new hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        super().set_params(**params)

        # Regenerate the circuit
        self._fm_str = self._generate_circuit_string()
        self._layered_encoding_circuit = self._build_layered_encoding_circuit()

        return self
