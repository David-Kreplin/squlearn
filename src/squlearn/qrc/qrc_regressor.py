import numpy as np
from typing import Union

from sklearn.base import RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase

from .base_qrc import BaseQRC
from ..util import Executor


class QRCRegressor(BaseQRC, RegressorMixin):
    """Quantum Reservoir Computing for regression.

    This class implements a Quantum Reservoir Computing (QRC) framework designed for
    regression tasks. In QRC, data is encoded into a quantum system—referred to as the
    quantum reservoir—using an encoding circuit. The state of the quantum reservoir is then
    measured using a set of quantum operators, often randomly chosen. The measured values,
    also known as expectation values, are used as features for a classical machine learning model,
    usually a simple linear regression, to perform the regression.

    Args:
        encoding_circuit (EncodingCircuitBase): The encoding circuit to use for encoding the data
            into the reservoir.
        executor (Executor): Executor instance
        ml_model (str): The classical machine learning model to use (default: linear). Possible
            values are:

                * ``"mlp"`` for a multi-layer perceptron regression model.
                * ``"linear"`` for a linear regression model.
                * ``"kernel"`` for a kernel ridge regression  with a linear kernel.

        ml_model_options (dict): The options for the machine learning model. Default options of the
            sklearn model are used if None.
        operators (Union[ObservableBase, list[ObservableBase], str]): Strategy for generating the
            operators used to measure the quantum reservoir. Possible values are:

                * ``"random_paulis"`` generates random Pauli operators (default).
                * ``"single_paulis"`` generates single qubit Pauli operators.

            Alternatively, a list of ObservableBase objects can be provided.
        num_operators (int): The number of random Pauli operators to generate for
            ``"operators = random_paulis"`` (default: 100).
        operator_seed (int): The seed for the random operator generation for
            ``"operators = random_paulis"`` (default: 0).
        param_ini (Union[np.ndarray, None]): The parameters for the encoding circuit.
        param_op_ini (Union[np.ndarray, None]): The initial parameters for the operators.
        parameter_seed (Union[int, None]): The seed for the initial parameter generation if no
            parameters are given.
        caching (bool): Whether to cache the results of the evaluated expectation values.

    See Also
    --------
        squlearn.qrc.QRCClassifier: Quantum Reservoir Computing for Classification.
        squlearn.qrc.base_qrc.BaseQRC: Base class for Quantum Reservoir Computing.

    **Example: Fitting the logarithm function with Quantum Reservoir Computing**

    .. code-block:: python

        import numpy as np
        from squlearn import Executor
        from squlearn.encoding_circuit import HubregtsenEncodingCircuit
        from squlearn.qrc import QRCRegressor
        from sklearn.model_selection import train_test_split

        X, y = np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)

        reg = QRCRegressor(HubregtsenEncodingCircuit(num_qubits=4, num_features=1),
                            Executor(),
                            ml_model="linear",
                            operators="random_paulis",
                            num_operators=300,
                            )

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

    Methods:
    --------
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        ml_model: str = "linear",
        ml_model_options: Union[dict, None] = None,
        operators: Union[ObservableBase, list[ObservableBase], str] = "random_paulis",
        num_operators: int = 100,
        operator_seed: int = 0,
        param_ini: Union[np.ndarray, None] = None,
        param_op_ini: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        caching: bool = True,
    ) -> None:

        super().__init__(
            encoding_circuit=encoding_circuit,
            executor=executor,
            ml_model=ml_model,
            ml_model_options=ml_model_options,
            operators=operators,
            num_operators=num_operators,
            operator_seed=operator_seed,
            param_ini=param_ini,
            param_op_ini=param_op_ini,
            parameter_seed=parameter_seed,
            caching=caching,
        )

    def _initialize_ml_model(self):

        if self.ml_model_options is None:
            self.ml_model_options = {}

        if self.ml_model == "mlp":
            self._ml_model = MLPRegressor(**self.ml_model_options)
        elif self.ml_model == "linear":
            self._ml_model = LinearRegression(**self.ml_model_options)
        elif self.ml_model == "kernel":
            self._ml_model = KernelRidge(**self.ml_model_options)
        else:
            raise ValueError("Invalid ml_model. Please choose 'mlp', 'linear' or 'kernel'.")
