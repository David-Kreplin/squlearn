from qiskit.primitives import BaseSampler
from qiskit.providers import Options
from qiskit_ibm_runtime.options import Options as qiskit_ibm_runtime_Options
import copy
from typing import Union, Callable, Dict, Optional, Any, List, Tuple

from dataclasses import asdict
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.base import SamplerResult
from qiskit.providers import JobV1 as Job
from qiskit import Aer, transpile
from qiskit.primitives import Sampler as qiskit_primitives_Sampler
from qiskit.primitives import BackendSampler as qiskit_primitives_BackendSampler
from qiskit_ibm_runtime import Sampler as qiskit_ibm_runtime_Sampler
from qiskit_ibm_runtime import RuntimeJob
from qiskit.primitives.utils import _circuit_key
from qiskit.primitives.primitive_job import PrimitiveJob


def custom_result_method(self):
    return self._result


class ParallelSampler(BaseSampler):
    """
    Special Sampler Primitive that parallelize circuits to be passed to the sampler.

    Args:
        sampler (BaseSampler): The sampler instance to use
        num_parallel (int, optional): The number of times the circuit is duplicated. Defaults to None, which means automatic determination.
        transpiler (callable, optional): A function for transpiling quantum circuits. Defaults to a standard transpile function if not provided.
        options (Options or qiskit_ibm_runtime_Options, optional): Configuration settings for the instance.

    """

    def __init__(
        self,
        sampler: BaseSampler,
        num_parallel: Optional[int] = None,
        transpiler: Optional[Callable] = None,
        options: Optional[Union[Options, qiskit_ibm_runtime_Options, Any]] = None,
    ) -> None:
        if isinstance(options, Options) or isinstance(options, qiskit_ibm_runtime_Options):
            try:
                options_ini = copy.deepcopy(options).__dict__
            except Exception:
                options_ini = asdict(copy.deepcopy(options))
        else:
            options_ini = options

        super().__init__(options=options_ini)
        self._sampler = sampler
        self._num_parallel = num_parallel
        if transpiler is None:
            self._transpiler = transpile
        else:
            self._transpiler = transpiler
        self.check_sampler()
        self._cache = {}

    def check_sampler(self) -> None:
        """Configures the backend and shot settings based on the properties of the provided sampler object."""
        self.shots = None
        if hasattr(self._sampler.options, "execution"):
            self.shots = self._sampler.options.get("execution").get("shots", None)

        if isinstance(self._sampler, qiskit_primitives_Sampler):
            # this is only a hack, there is no real backend in the Primitive Sampler class
            self._backend = Aer.get_backend("statevector_simulator")
        elif isinstance(self._sampler, qiskit_primitives_BackendSampler):
            self._backend = self._sampler._backend
            shots_sampler = self._sampler.options.get("shots", 0)
            if shots_sampler == 0:
                if self.shots is None:
                    self.shots = 1024
                self._sampler.set_options(shots=self.shots)
            else:
                self.shots = shots_sampler
        # Real Backend
        elif hasattr(self._sampler, "session"):
            self._backend = self._sampler.session.service.get_backend(
                self._sampler.session.backend()
            )
            self._session_active = True
        else:
            raise RuntimeError("No backend found in the given Sampler Primitive!")

    def set_shots(self, num_shots: Union[int, None]) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int or None): Number of shots that are set
        """
        self._shots = num_shots

        if num_shots is None:
            self._logger.info("Set shots to {}".format(num_shots))
            num_shots = 0

        # Update shots in backend
        if self._backend is not None:
            if "statevector_simulator" not in str(self._backend):
                self._backend.options.shots = num_shots

        # Update shots in sampler primitive
        if self._sampler is not None:
            if isinstance(self._sampler, qiskit_primitives_Sampler):
                if num_shots == 0:
                    self._sampler.set_options(shots=None)
                else:
                    self._sampler.set_options(shots=num_shots)
                try:
                    self._options_sampler["shots"] = num_shots
                except Exception:
                    pass  # no option available
            elif isinstance(self._sampler, qiskit_primitives_BackendSampler):
                self._sampler.set_options(shots=num_shots)
                try:
                    self._options_sampler["shots"] = num_shots
                except Exception:
                    pass  # no option available
            elif isinstance(self._sampler, qiskit_ibm_runtime_Sampler):
                execution = self._sampler.options.get("execution")
                execution["shots"] = num_shots
                self._sampler.set_options(execution=execution)
                try:
                    self._options_sampler["execution"]["shots"] = num_shots
                except Exception:
                    pass  # no options_sampler or no execution in options_sampler
            else:
                raise RuntimeError("Unknown sampler type!")

    def recover_original_distribution(
        self,
        duplicated_result: Dict,
        total_qubits: int,
        original_qubits: int,
        qubit_mapping: Optional[Dict] = None,
    ) -> Dict:
        """
        Recover the original probability distribution from the results of a duplicated quantum circuit.

        This function processes the measurement results from a duplicated quantum circuit and extracts
        the probability distribution of the original circuit. It supports cases where the duplicated
        circuits are mapped to adjacent and ordered qubits, as well as cases with a specific qubit mapping.

        Args:
            duplicated_result (dict): A dictionary with keys as integers representing the outcomes of the
                                    duplicated circuit and values as their respective probabilities.
            total_qubits (int): The total number of qubits in the duplicated circuit.
            original_qubits (int): The number of qubits in the original circuit.
            qubit_mapping (list, optional): A list of integers representing the specific mapping of qubits
                                            from the original to the duplicated circuit. If not provided,
                                            it's assumed that the duplications are adjacent and ordered.

        Returns:
            dict: A dictionary representing the recovered probability distribution of the original circuit.
            The keys are integers representing the outcomes, and the values are the probabilities.

        """
        # If no mapping is provided, assume duplications are adjacent and ordered
        # Define the qubit mapping if not provided
        if qubit_mapping is None:
            qubit_mapping = list(range(total_qubits))

        # Calculate the number of duplications
        n_duplications = len(qubit_mapping) // original_qubits

        # Initialize the original distribution dictionary
        original_distribution = {i: 0 for i in range(2**original_qubits)}

        # Process each outcome in the duplicated results
        for outcome, probability in duplicated_result.items():
            # Convert to binary and reverse for LSB convention
            binary_outcome = format(outcome, f"0{total_qubits}b")[::-1]

            # Initialize a list for the reordered outcome
            reordered_outcome = ["0"] * total_qubits

            # Reorder based on qubit mapping
            for qubit_index in qubit_mapping:
                reordered_outcome[qubit_index] = binary_outcome[qubit_mapping.index(qubit_index)]

            # Join reordered outcome to a string
            reordered_outcome_str = "".join(reordered_outcome)[: n_duplications * original_qubits]

            # Split and aggregate probabilities for each duplicated circuit
            for i in range(n_duplications):
                part_start = i * original_qubits
                part_end = part_start + original_qubits
                part_outcome = int(
                    reordered_outcome_str[part_start:part_end][::-1], 2
                )  # Reverse back to original order
                original_distribution[part_outcome] += probability / n_duplications

        # Normalize the distribution
        total_probability = sum(original_distribution.values())
        for outcome in original_distribution:
            original_distribution[outcome] /= total_probability

        return original_distribution

    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        parameter_values: Union[List[float], List[List[float]]] = None,
        process_result: Optional[bool] = True,
        **run_options,
    ) -> Job:
        """
        Executes quantum circuits with duplications, and result processing.

        This method overwrites the sampler primitive run method, adding functionalities like circuit duplication and result processing. It utilizes the Executor class for automatic session handling.

        Args:
            circuits (Union[QuantumCircuit, List[QuantumCircuit]]): A single QuantumCircuit or a list of QuantumCircuits to be executed.
            parameter_values (Optional[Union[List[float], List[List[float]]]], optional): Values for parameterized circuits. Can be a single list of values or a list of lists for multiple circuits. Defaults to None.
            n_duplications (Optional[int], optional): The number of times the circuit execution is to be duplicated. Useful for certain types of quantum simulations. Defaults to None.
            process_result (Optional[bool], optional): If True, the results will be processed in a specific manner as defined within the method. Defaults to True.
            **run_options: Additional keyword arguments that are passed to the execution method. These options are specific to the implementation and execution environment.

        Returns:
            Job: An object representing the job submitted to the quantum backend for execution. The job object can be used to query for the status and result of the execution.
        """
        dupl_circuits = []
        self.n_dupl_list = []

        if not isinstance(circuits, list):
            circuits = [circuits]

        if "shots" in run_options:
            self.shots = run_options["shots"]
            run_options.pop("shots")

        for circ in circuits:
            duplicated_circ, self._num_parallel = self.create_mapped_circuit(
                circ, num_parallel=self._num_parallel, return_duplications=True
            )
            dupl_circuits.append(duplicated_circ)
            self.n_dupl_list.append(self._num_parallel)

        result_job = self._sampler.run(
            circuits=dupl_circuits,
            parameter_values=parameter_values,
            **run_options,
        )

        result = result_job.result()
        for meta in result.metadata:
            meta["shots"] = self.shots

        if process_result:
            result = self.process_result(result, circuits, duplicated_circ.num_qubits)

        result_job._result = result
        result_job.result = custom_result_method.__get__(result_job, type(result_job))
        return result_job

    def process_result(
        self,
        result_job: Union[RuntimeJob, PrimitiveJob, Job],
        original_circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        total_qubits: int,
        qubit_mapping: Optional[List] = None,
    ) -> Job:
        """
        Processes the result of a quantum job to map it to the distribution of the original circuits.

        This method takes the result from a quantum job and adjusts it to correspond with the qubit distribution of the original circuits.

        Args:
            result_job (Union[RuntimeJob, PrimitiveJob, Job]): The job object containing the results to be processed. This can be a RuntimeJob, PrimitiveJob, or a standard Job.
            original_circuits (Union[QuantumCircuit, List[QuantumCircuit]]): The original quantum circuit or a list of quantum circuits for which the results are intended.
            total_qubits (int): The total number of qubits in the modified circuit execution.
            qubit_mapping (Optional[List], optional): A list that maps the modified qubits back to the original qubits. Defaults to None.

        Returns:
            Job: The modified job object with the result now reflecting the original circuit distribution.
            If the input was not a Job instance, the processed result is returned directly.
        """
        wrap_result = False
        if isinstance(result_job, (RuntimeJob, PrimitiveJob, Job)):
            result = result_job.result()
            wrap_result = True
        else:
            result = result_job

        if not isinstance(original_circuits, list):
            original_circuits = [original_circuits]

        for ii, circ in enumerate(original_circuits):
            num_qubits = circ.num_qubits
            original_dist = self.recover_original_distribution(
                result.quasi_dists[ii], total_qubits, num_qubits, qubit_mapping
            )
            result.quasi_dists[ii] = original_dist

        if wrap_result:
            result_job._result = result
            result_job.result = custom_result_method.__get__(result_job, type(result_job))
            result = result_job
        return result

    def _run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        parameter_values: Union[List[float], List[List[float]]] = None,
        **run_options,
    ) -> Job:
        """
        Overwrites the sampler primitive run method, to evaluate circuits.

        Input arguments are the same as in Qiskit's sampler.run()

        """
        return self._sampler._run(
            circuits=circuits,
            parameter_values=parameter_values,
            **run_options,
        )

    def _call(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        parameter_values: Union[List[float], List[List[float]]] = None,
        **run_options,
    ) -> SamplerResult:
        """Has to be passed through, otherwise python will complain about the abstract method"""
        return self._sampler._call(circuits, parameter_values, **run_options)

    @property
    def circuits(self):
        """Quantum circuits to be sampled.

        Returns:
            The quantum circuits to be sampled.
        """
        return tuple(self._sampler.circuits)

    @property
    def Args(self):
        """Args of quantum circuits.

        Returns:
            List of the Args in each quantum circuit.
        """
        return tuple(self._sampler.Args)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._sampler.options

    def set_options(self, **fields) -> None:
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._sampler.set_options(**fields)
        self._sampler._options_sampler = self._sampler.options

    def create_mapped_circuit(
        self,
        circuit: QuantumCircuit,
        num_parallel: Optional[int] = None,
        return_duplications: Optional[bool] = False,
        max_qubits: Optional[int] = None,
    ) -> Union[QuantumCircuit, Tuple[QuantumCircuit, int]]:
        """
        Maps a given quantum circuit, optionally duplicating it to fill the backend capacity.

        This method maps the provided quantum circuit, potentially duplicating it to utilize as much of the backend's capacity as possible.
        The duplication is controlled by the 'n_duplications' or 'max_qubits' Args.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be mapped.
            n_duplications (Optional[int], optional): Specifies the number of times the circuit should be duplicated. Defaults to None.
            return_duplications (Optional[bool], optional): If True, returns a tuple of the mapped circuit and the number of duplications. Defaults to False.
            max_qubits (Optional[int], optional): The maximum number of qubits to use from the backend. Defaults to the number of qubits in the backend if None.

        Returns:
            Union[QuantumCircuit, Tuple[QuantumCircuit, int]]: The mapped quantum circuit.
            If 'return_duplications' is True, returns a tuple containing the mapped circuit and the number of duplications.

        Raises:
            Warning: If no number of qubits is found in the given Sampler Primitive.
            ValueError: If the total number of qubits required for duplications exceeds the backend's capacity.
        """

        if max_qubits is None:
            max_qubits = self._backend.configuration().n_qubits
            if max_qubits is None:
                raise Warning("No number of qubits found in the given Sampler Primitive!")

        print("max_qubits",max_qubits)

        # check that n_duplication is None, i.e. not provided.
        if num_parallel is None:
            num_parallel = int(max_qubits // circuit.num_qubits)

        if num_parallel * circuit.num_qubits > max_qubits:
            raise ValueError(
                f"The number of qubits in the circuit ({circuit.num_qubits}) * n_duplications ({num_parallel}) "
                f"is greater than the total number of qubits in the backend ({max_qubits})"
            )

        # create the circuit
        mapped_circuit = circuit.copy()

        # duplicate the circuit
        for _ in range(num_parallel - 1):
            mapped_circuit.tensor(circuit, inplace=True)

        shots = self.shots
        if shots is None:
            shots = 0
        print(
            f"\nCircuit with {circuit.num_qubits} qubits has been duplicated {num_parallel} times."
            f"\nReducing shots from {shots} to {int(shots/num_parallel)}"
            f"\nBackend has {max_qubits} qubits."
        )

        if self.shots is not None:
            self.set_shots(int(self.shots / num_parallel))
        if return_duplications:
            return mapped_circuit, num_parallel
        else:
            return mapped_circuit

    def transpile(self, circuit: QuantumCircuit, **options) -> QuantumCircuit:
        """
        Transpiles a given quantum circuit, using cached results if available.

        This method checks if the provided circuit has already been transpiled by looking it up in a cache.
        If it's in the cache, the cached transpiled circuit is returned.
        Otherwise, the circuit is transpiled using the provided transpiler function, and the result is cached for future use.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be transpiled.
            **options: Additional keyword arguments for the transpiler. These options allow for customization of the transpilation process.

        Returns:
            The transpiled quantum circuit.
        """
        key = _circuit_key(circuit)
        if key in self._cache:
            transpiled_circuit = self._cache[key]
        else:
            transpiled_circuit = self._transpiler(circuit, **options)
            self._cache[key] = transpiled_circuit
        return transpiled_circuit
