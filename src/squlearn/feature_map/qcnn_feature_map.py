
import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_gate

from .feature_map_base import FeatureMapBase


class operation():
    def __init__(self, layer : str, QC : QuantumCircuit, entangled : bool, operator : Union[str,None], var_param : bool, target_qubit : Union[int,list,None], closed : bool): #TODO: siehe auch convouliton: (...entangled: bool, var_param, operator)
        """
        Args:
            layer[str]: name of the layer (default: "c", "p" or "f")
            QC[QuantumCircuit]: a quantum circuit by qiskit
            entangled[bool]: only for convolution layers: Stores, if the operation is entangled
            operator[str,None]: Name of the used operator 
            var_param[bool]: Stores, if the parameters get bigger by every gate (True) or only per operation (False)
            target_qubit[int,list,None]: only for pooling layers: Target qubit, which will be left for the next gates; 
                                        only for convolutional layers: If target qubit is a list, a single gate is applied on qubits, which are in the list
            closed[bool]: Only for entangled convolutional layers: If true, some gates can control first and last qubits and skip qubits in between. If false, gates are only applied on successive qubits.
        """
        self.layer = layer
        self.QC = QC
        if operator == None:
            if layer == "c":
                self.operator = "U"
            elif layer == "p":
                self.operator = "V"
            else:
                self.operator = "F"
        else:
            self.operator = operator
        self.var_param = var_param
        # not necessary for pooling and fully connected layers:
        self.entangled = entangled
        self.closed = closed
        # only necessary for pooling and convolutional layers:
        self.target_qubit = target_qubit
        #------------------------------------------------------------------------------------------------------------------------------------------
        # only necessary for reversed build up of the circuit (if num_qubits is not given):
        # reversed_input_qubits are the qubits on the right side of the gate and reversed_output_qubits the qubits coming out of the left side.
        self.reversed_input_qubits = 0 #has to be overwritten in get_circuit by fully_layer, convolution_layer or pooling_layer
        self.reversed_output_qubits = 0 #has to be overwritten


class qcnn_feature_map(FeatureMapBase):
    """
    A class for a Quantum Convolutional Neural Network (qcnn) feature map. 
    A qcnn is based on a regular Convolutional Neural Network (cnn) with convolutional layers, pooling layers and fully connected layers.
    """

    def __init__(self, num_qubits: Union[int,None] = None):
        """
        Attributes:
            _num_qubits[int,None]: Number of qubits used in the whole feature map. If None, the circuit will count it itself by reversed iterating through the operation_list.
            operation_list[list]: List of operations, which contains convolutional layers, pooling layers and fully connected layers.
            parameter_counter[int]: Stores, how many parameters are used in the whole feature map. Only if num_qubits is an integer.
            controlled_qubits[int]: Stores, how many qubits are left to be controlled (with pooling layers there are less qubits left to control as before). Only if num qubits is an integer.
            fully_connected_layer[bool]: Stores, if a fully connected layer is already in the operation_list
        """
        self._num_qubits = num_qubits
        self.operation_list = []
        self.parameter_counter = 0
        self.controlled_qubits = num_qubits
        self.fully_connected_layer = False


    def _add_operation(self, operation:operation):
        """
        adds an operation to the operation_list and increases the parameter counter parameter_counter (if self._num_qubits is not None).
        Args:
            operation[operation]: An operation of the class operation, which contains a convolutional layer, pooling layer or a fully connected layer.
        """
        if operation.layer == "f":
            if self.fully_connected_layer:
                raise TypeError("There must be at most one fully connected layer.")
            else:
                self.fully_connected_layer = True

        gate_num_qubits = operation.QC.num_qubits # stores, how many qubits are addressed by the gate 
        if (self.controlled_qubits != None) and (gate_num_qubits > self.controlled_qubits):
            if operation.layer == "c":
                print("Warning on convolutional layer: The quantum circuit input controls too many qubits:",gate_num_qubits,"qubits on input vs.",self.controlled_qubits,"qubits on the actual circuit.")
            elif operation.layer == "p":
                print("Warning on pooling layer: The quantum circuit input controls too many qubits:",gate_num_qubits,"qubits on input vs.",self.controlled_qubits,"qubits on the actual circuit.")
            else:
                print("Warning on fully connected layer: The quantum circuit input controls too many qubits:",gate_num_qubits,"qubits on input vs.",self.controlled_qubits,"qubits on the actual circuit.")
        else:
            gate_params = operation.QC.parameters # stores, which parameter vectors are used by the gate
            if self.controlled_qubits != None: 
                # If controlled qubits are None, do nothing but adding the operation to the operation_list, 
                # otherwise increase parameter_counter, adjust the controlled qubits depending on the operation.
                if operation.var_param:
                    if operation.layer == "c":
                        if operation.target_qubit != None:
                            self.parameter_counter += len(gate_params)
                        else:
                            if operation.entangled:
                                if operation.closed:
                                    num_gates = gate_num_qubits * int(self.num_qubits/gate_num_qubits) # stores, how often a gate is applied to the circuit
                                else:
                                    num_gates = 0
                                    for i in range(gate_num_qubits):
                                        for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                            if self.num_qubits-first_gate_qubit >= gate_num_qubits:
                                                num_gates += 1
                            else:
                                num_gates = int(self.controlled_qubits/gate_num_qubits)
                            self.parameter_counter += num_gates * len(gate_params)
                    elif operation.layer == "p":
                        qubits_untouched = self.controlled_qubits%gate_num_qubits
                        num_gates = int(self.controlled_qubits/gate_num_qubits)
                        self.controlled_qubits = num_gates + qubits_untouched
                        self.parameter_counter += num_gates * len(gate_params)
                    else:
                        self.parameter_counter += len(gate_params)
                else:
                    if operation.layer == "p":
                        qubits_untouched = self.controlled_qubits%gate_num_qubits
                        num_gates = int(self.controlled_qubits/gate_num_qubits)
                        self.controlled_qubits = num_gates + qubits_untouched
                    self.parameter_counter += len(gate_params)
            self.operation_list.append(operation)

    @property
    def num_qubits(self) -> int:
        """ Returns the number of qubits of the feature map. """
        if self._num_qubits == None:
            num_qubits = 1
            for operation in reversed(self.operation_list):
                gate_qubits = operation.QC.num_qubits
                if operation.layer == "f":
                    num_qubits = gate_qubits
                elif operation.layer == "p":
                    num_qubits = gate_qubits*num_qubits
            return num_qubits
        else:
            return self._num_qubits

    @ property
    def num_features(self) -> int:
        """ Returns the dimension of features of the feature map. """
        return self.num_qubits 

    @ property
    def num_parameters(self) -> int:
        """ Returns the number of trainable parameters of the feature map. """
        if self._num_qubits == None: 
            # If _num_qubits are not known (None), calculate the parameters using the operation_list by reversed iteration.
            param_reversed_counter = 0
            num_qubits = 1
            for operation in reversed(self.operation_list):
                gate_num_qubits = operation.QC.num_qubits
                gate_params = operation.QC.parameters
                if operation.layer == "f":
                    param_reversed_counter = len(gate_params)
                    num_qubits = gate_num_qubits
                elif operation.layer == "c":
                    if operation.target_qubit == None:
                        if operation.var_param:
                            if operation.entangled:
                                gate_block_appearance = int(num_qubits/gate_num_qubits)
                                if operation.closed:
                                    gate_appearance = gate_block_appearance * gate_num_qubits
                                else:
                                    gate_appearance = 0
                                    for i in range(gate_num_qubits): # on entangling we need exactly m=gate_num_qubits layers 
                                        gate_block_appearance = int((num_qubits-i)/gate_num_qubits)
                                        gate_appearance += gate_block_appearance
                            else:
                                gate_appearance = int(num_qubits/gate_num_qubits)
                            param_reversed_counter += gate_appearance*len(gate_params)
                        else:
                            param_reversed_counter += len(gate_params)
                    else: # TODO: option einer Liste vielleicht weglassen, da var_param und entangled an der Stelle sowieso keine Bedeutung haben
                        param_reversed_counter += len(gate_params)
                elif operation.layer == "p":
                    if operation.var_param:
                        gate_appearance = num_qubits
                        num_qubits = gate_num_qubits*num_qubits
                        param_reversed_counter += gate_appearance*len(gate_params)
                    else:
                        num_qubits = gate_num_qubits*num_qubits
                        param_reversed_counter += len(gate_params)
                else:
                    raise NameError("Unknown operation layer.")
            return param_reversed_counter
        else:
            return self.parameter_counter
    
    def get_circuit(self,
                    features: Union[ParameterVector,np.ndarray],
                    parameters: Union[ParameterVector,np.ndarray]
                    ) -> QuantumCircuit:
        """
        Return the circuit feature map
        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in qiskit QuantumCircuit format
        """
        if self._num_qubits == None:
           return self.get_circuit_without_qubits(features,parameters)
        else:
           return self.get_circuit_with_qubits(features,parameters)
    
    def get_circuit_with_qubits(self,
                    features: Union[ParameterVector,np.ndarray],
                    parameters: Union[ParameterVector,np.ndarray]
                    ) -> QuantumCircuit:
        """
        Builds and Returns the quantum circuit of the qcnn feature map. Is used, if the user gives the number of qubits.
        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            parameters Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained
        Return:
            returns the quantum circuit of the qcnn feature map.
        """
        QC = QuantumCircuit(self.num_qubits)
        controlled_qubits = [i for i in range(self.num_qubits)]
        label_name_dict = {"c":0,"p":0} # is required for counting the name of the operator, if the user doesn't defines it by himself
        global_param_counter = 0 # is required for assigning the local parameters (in every gate) to the global parameter
        for operation in self.operation_list:
            if len(controlled_qubits) == 0:
                raise IndexError("There are to many pooling layers, because there are no qubits left.")
            gate_num_qubits = operation.QC.num_qubits # with that, the algorithm knows, how many qubits are addressed by the gate
            gate_params = operation.QC.parameters # stores, which parameter vectors are used by the gate
            if operation.layer == "c":
                if operation.target_qubit == None:
                    if operation.entangled:
                        if operation.closed:
                            if operation.var_param:
                                num_gates = gate_num_qubits * int(self.num_qubits/gate_num_qubits)
                                # To store every map for every gate: # [{x1:y1,x2:y2},{x1:y3,x2:y4}, etc.]:
                                map_dict_list = [{gate_params[i]:parameters[i+k*len(gate_params)] for i in range(len(gate_params))} for k in range(num_gates)]
                                global_param_counter += num_gates * len(gate_params)
                                map_dict_iter = 0
                                # Testing if num_qubits is dividable by gate_qubits, if so, iterate until every qubit is touched, 
                                # otherwise iterate until there are not enough qubits left for the gate without touching the first qubit (and the last qubit) again:
                                if self.num_qubits%gate_num_qubits == 0:
                                    for i in range(gate_num_qubits):
                                        for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict_list[map_dict_iter],label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                            # stores the qubits, on which the gate operates; 
                                            # with modulo number of qubits, so there can be gates which operate e.g. on the last and the first qubit simultaneously (and skips the rest):
                                            gate_on_qubits = [t%self.num_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                            QC.append(gate, gate_on_qubits)
                                            map_dict_iter += 1
                                else:
                                    gates_to_place_per_block = int(self.num_qubits/gate_num_qubits)
                                    for i in range(gate_num_qubits):
                                        gates_placed = 0 # counts how many gates (vertically) are placed in the ith block
                                        enough_gate_placed = False # stores if there are enough gates placed in the ith iteration, if not, force the first iterations to place as many gates as required, which is stored in gates_to_place_per_block,
                                        #this is important in the case num_qubits < 2*gate_qubits. In that case there must be gates, which control the first and the last qubit simultaneously (and skips the rest).
                                        for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                            if (self.num_qubits-first_gate_qubit>= gate_num_qubits) or not enough_gate_placed:
                                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict_list[map_dict_iter],label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                                # stores the qubits, on which the gate operates; 
                                                # with modulo number of qubits, so there can be gates which operate e.g. on the last and the first qubit simultaneously (and skips the rest):
                                                gate_on_qubits = [t%self.num_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                                QC.append(gate, gate_on_qubits)
                                                map_dict_iter += 1
                                                gates_placed += 1
                                                if gates_to_place_per_block == gates_placed:
                                                    enough_gate_placed = True
                            else:
                                map_dict = {gate_params[i]:parameters[i] for i in range(len(gate_params))}
                                global_param_counter += len(gate_params)
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                # Testing if num_qubits is dividable by gate_qubits, if so, iterate until every qubit is touched, 
                                # otherwise iterate until there are not enough qubits left for the gate without touching the first qubit (and the last qubit) again:
                                if self.num_qubits%gate_num_qubits == 0:
                                    for i in range(gate_num_qubits):
                                        for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                            gate_on_qubits = [t%self.num_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                            QC.append(gate,gate_on_qubits)
                                else:
                                    gates_to_place_per_block = int(self.num_qubits/gate_num_qubits)
                                    for i in range(gate_num_qubits):
                                        gates_placed = 0 # counts how many gates (vertically) are placed in the ith block
                                        enough_gate_placed = False # stores if there are enough gates placed in the ith iteration, if not, force the first iterations to place as many gates as required, which is stored in gates_to_place_per_block
                                        #this is important in the case num_qubits < 2*gate_qubits. In that case there must be gates, which control the first and the last qubit simultaneously.
                                        for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                            if (self.num_qubits-first_gate_qubit >= gate_num_qubits) or not enough_gate_placed:
                                                gate_on_qubits = [t%self.num_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                                QC.append(gate,gate_on_qubits)
                                                gates_placed += 1
                                                if gates_to_place_per_block == gates_placed:
                                                    enough_gate_placed = True
                        else:
                            if operation.var_param:
                                num_gates = 0
                                for i in range(gate_num_qubits):
                                    for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                        # apply a gate only, if there is enough space at the bottom (e.g. gate with 3 qubits, but there are only two qubits left at the bottom, so no gate is applied)
                                        if self.num_qubits-first_gate_qubit >= gate_num_qubits:
                                            num_gates += 1
                                map_dict_list = [{gate_params[i]:parameters[i+k*len(gate_params)] for i in range(len(gate_params))} for k in range(num_gates)]
                                global_param_counter += num_gates * len(gate_params)
                                map_dict_iter = 0
                                for i in range(gate_num_qubits):
                                    for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                        # apply a gate only, if there is enough space at the bottom (e.g. gate with 3 qubits, but there are only two qubits left at the bottom, so no gate is applied)
                                        if self.num_qubits-first_gate_qubit >= gate_num_qubits:
                                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict_list[map_dict_iter],label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                            gate_on_qubits = range(first_gate_qubit,first_gate_qubit+gate_num_qubits)
                                            QC.append(gate,gate_on_qubits)
                                            map_dict_iter += 1
                                # Attention: With 3 or more qubits it can happen , that the gate layer doesn't reach every qubit, e.g.:
                                # we have a system with 5 qubits and a gate supplies to 3 qubits. Than the gate layer do not reach qubit 3 and 4 (if counting begins at 0). 
                            else:
                                map_dict = {gate_params[i]:parameters[i] for i in range(len(gate_params))}
                                global_param_counter += len(gate_params)
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                for i in range(gate_num_qubits):
                                    for first_gate_qubit in range(i, self.num_qubits, gate_num_qubits):
                                        # apply a gate only, if there is enough space at the bottom (e.g. gate with 3 qubits, but there are only two qubits left at the bottom, so no gate is applied)
                                        if self.num_qubits-first_gate_qubit >= gate_num_qubits:
                                            gate_on_qubits = range(first_gate_qubit,first_gate_qubit+gate_num_qubits)
                                            QC.append(gate,gate_on_qubits)
                                # Attention: With 3 or more qubits it can happen , that the gate layer doesn't reach every qubit, e.g.:
                                # we have a system with 5 qubits and a gate supplies to 3 qubits. Than the gate layer do not reach qubit 3 and 4 (if counting begins at 0). 

                elif type(operation.target_qubit) == list: 
                    #apply only one gate by using a list of the desired qubits on that qubits
                    num_params = len(gate_params)
                    map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(num_params)}
                    global_param_counter += num_params
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                    gate_on_qubits = operation.target_qubit
                    QC.append(gate,gate_on_qubits)
                    
                else: 
                    # Target qubit is an integer, so the first controlled qubit will be the target qubit; example: Gate controls 3 qubits, target qubit is 2, so the gate controls qubit 2,3 and 4
                    num_params = len(gate_params)
                    map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(num_params)}
                    global_param_counter += num_params
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                    gate_on_qubits = []
                    for k in range(operation.target_qubit,operation.target_qubit+gate_num_qubits):
                        gate_on_qubits.append(controlled_qubits[k])
                    QC.append(gate, gate_on_qubits)
                operator_number = label_name_dict["c"]
                label_name_dict.update({"c":operator_number+1})

            elif operation.layer == "p": # Default: last qubit is always the target qubit
                if len(controlled_qubits)<gate_num_qubits:
                    print("Warning: This pooling operation will not be executed, because there are not enough qubits left.")
                else:
                    if operation.target_qubit == None:
                        if operation.var_param:
                            num_gates = int(len(controlled_qubits)/gate_num_qubits) # stores, how often a gate is applied to the circuit
                            num_params = num_gates * len(gate_params) # stores, how many parameters are used in this operation
                            # To store every map for every gate:
                            map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(num_gates)] # TODO: could be more efficient, if one build gate in for loop without the list
                            global_param_counter += num_params
                            map_iter = 0
                            new_controlled_qubits = []
                            last_qubit = 0
                            for i,j in zip(range(len(controlled_qubits))[0::gate_num_qubits], controlled_qubits[gate_num_qubits-1::gate_num_qubits]):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                                QC.append(gate, controlled_qubits[i:i+gate_num_qubits])
                                #new_controlled_qubits.append(controlled_qubits[i])
                                new_controlled_qubits.append(j)
                                last_qubit = j                                                             
                        else:
                            map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                            global_param_counter += len(gate_params)
                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                            new_controlled_qubits = []
                            # Works with gates of length >= 2:
                            last_qubit = 0
                            for i,j in zip(range(len(controlled_qubits))[0::gate_num_qubits], controlled_qubits[gate_num_qubits-1::gate_num_qubits]):
                                QC.append(gate, controlled_qubits[i:i+gate_num_qubits])
                                new_controlled_qubits.append(j)
                                last_qubit = j #The last (lowest) qubit, which is touched by the pooling layer
                        # checks how many qubits are untouched by pooling gates and adds them to the controlled qubits list by checking the qubits under the last (lowest) qubit:
                        last_qubit_index = controlled_qubits.index(last_qubit)
                        for i in controlled_qubits[last_qubit_index+1:]:
                            new_controlled_qubits.append(i)
                        controlled_qubits = new_controlled_qubits
                        operator_number = label_name_dict["p"]
                        label_name_dict.update({"p":operator_number+1})
                    else: # operation.target_qubit is an integer, so the user chose the remaining target qubit                    
                        if operation.var_param:
                            num_gates = int(len(controlled_qubits)/gate_num_qubits) # stores, how often a gate is applied to the circuit
                            num_params = num_gates * len(gate_params) # stores, how many parameters are used in this operation
                            # To store every map for every gate:
                            map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(num_gates)] # TODO: could be more efficient, if one build gate in for loop without the list
                            global_param_counter += num_params
                            map_iter = 0
                            new_controlled_qubits = []
                            remaining_qubit = operation.target_qubit # über die if-Schleife setzen
                            last_qubit = 0
                            for i,j in zip(range(len(controlled_qubits))[0::gate_num_qubits], controlled_qubits[gate_num_qubits-1::gate_num_qubits]):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                                QC.append(gate, controlled_qubits[i:i+gate_num_qubits])
                                new_controlled_qubits.append(controlled_qubits[i+remaining_qubit])
                                last_qubit = j
                        else:
                            map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                            global_param_counter += len(gate_params)
                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                            new_controlled_qubits = []
                            remaining_qubit = operation.target_qubit
                            # Works with gates of length \geq 2:
                            last_qubit = 0
                            for i,j in zip(range(len(controlled_qubits))[0::gate_num_qubits], controlled_qubits[gate_num_qubits-1::gate_num_qubits]):
                                QC.append(gate, controlled_qubits[i:i+gate_num_qubits])
                                new_controlled_qubits.append(controlled_qubits[i+remaining_qubit])
                                last_qubit = j
                        # checks how many qubits are untouched by pooling gates and adds them to the controlled qubits list by checking the qubits under the last (lowest) qubit:
                        last_qubit_index = controlled_qubits.index(last_qubit)
                        for i in controlled_qubits[last_qubit_index+1:]:
                            new_controlled_qubits.append(i)
                        controlled_qubits = new_controlled_qubits
                        operator_number = label_name_dict["p"]
                        label_name_dict.update({"p":operator_number+1})

            else: # For fully-connected layer
                if operation.QC.num_qubits != len(controlled_qubits):
                    raise ValueError("Fully-connected gate size is not the same as the number of qubits, that are left. Gate size: "+str(operation.QC.num_qubits)+", number of qubits left: "+str(len(controlled_qubits)))
                map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                num_params = len(gate_params)
                global_param_counter += num_params
                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                QC.append(gate, [k for k in controlled_qubits])
        if global_param_counter < len(parameters):
            raise IndexError("Input parameter vector has a greater size as needed: Input paramvec: ", len(parameters), " size needed:", global_param_counter )
        return QC


    def get_circuit_without_qubits(self,
                                   features: Union[ParameterVector,np.ndarray],
                                   parameters: Union[ParameterVector,np.ndarray]
                                   ) -> QuantumCircuit:
        """
        Builds and Returns the quantum circuit of the qcnn feature map. Is used, if the user doesn't give the number of qubits.
        Args:
            features Union[ParameterVector,np.ndarray]: -
            parameters Union[ParameterVector,np.ndarray]: Parameter vectors by qiskit or numpy arrays which are treated like free variables.
        Return:
            returns the quantum circuit of the qcnn feature map.
        """
        QC = QuantumCircuit(1)
        param_iter = len(parameters)  #Example: 5 Parameters (0,1,2,3,4); on fully: 3 Parameters (2,3,4); 
        #that means param_iter is 5 and param_length is 3. fully_layer gets a parametervectorlist with entries [2,3,4], after that param_iter is 5-3 = 2.
        reversed_input_qubits = 1
        fully_exists = False # stores, if a fully connected layer exists (only for error exceptions, e.g. two fully layers are not provided)
        other_layer_before_fully = False #stores, if there are other layers before the fully connected gate. If so and there is an other fully gate, raise an error.
        for operation in reversed(self.operation_list): 
            param_length = len(operation.QC.parameters) # stores the number of applications of parameters of one operation
            if operation.layer == "f":
                fully_exists = True
                if other_layer_before_fully:
                    raise NameError("It's not allowed to put a fully connected layer before an other layer.")
                # parameters[(param_iter-param_length):param_iter] will only give that parameter vectors, which are required in that operation:
                # Build a new circuit by overwriting the one-qubit quantum circuit by the fully connected layer:
                QC = self._fully_layer(operation, parameters[(param_iter-param_length):param_iter])
                reversed_input_qubits = operation.reversed_output_qubits
            elif operation.layer == "p":
                if not fully_exists:
                    other_layer_before_fully = True
                operation.reversed_input_qubits = reversed_input_qubits
                gate_num_qubits = operation.QC.num_qubits
                gate_appearance = operation.reversed_input_qubits
                if operation.var_param:
                    param_length = param_length * gate_appearance
                # building up the list of qubits, which will be left
                # Example: there are four 3-qubit gates, so the first one has 2 as target qubit, the second the fifth as target qubit. The target qubit of the third gate is number 6 and the last taret qubit is 9:
                qubits_right_side = []
                active_qubit_iterator = -1
                for i in range(int((reversed_input_qubits+1)/2)): # 
                    active_qubit_iterator += gate_num_qubits
                    qubits_right_side.append(active_qubit_iterator)
                active_qubit_iterator += 1
                for i in range(int((reversed_input_qubits+1)/2),reversed_input_qubits):
                    qubits_right_side.append(active_qubit_iterator)
                    active_qubit_iterator += gate_num_qubits
                # parameters[(param_iter-param_length):param_iter] will only give that parameter vectors, which are required in that operation:
                # Compose the pooling circuit to the left side of the already existing circuit:
                QC = self._pooling_layer(operation, parameters[(param_iter-param_length):param_iter]).compose(QC,qubits_right_side)
                reversed_input_qubits = operation.reversed_output_qubits
            else:
                if not fully_exists:
                    other_layer_before_fully = True
                operation.reversed_input_qubits = reversed_input_qubits
                gate_num_qubits = operation.QC.num_qubits
                if operation.var_param:
                    if operation.entangled:
                        gate_block_appearance = int(reversed_input_qubits/gate_num_qubits)
                        if operation.closed:
                            gate_appearance = gate_block_appearance * gate_num_qubits
                        else:
                            gate_appearance = 0
                            for i in range(gate_num_qubits):
                                gate_block_appearance = int((reversed_input_qubits-i)/gate_num_qubits)
                                gate_appearance += gate_block_appearance
                    else:
                        gate_appearance = int(reversed_input_qubits/gate_num_qubits)
                    param_length = param_length * gate_appearance
                # parameters[(param_iter-param_length):param_iter] will only give that parameter vectors, which are required in that operation:
                # Compose the convolutional circuit to the left side of the already existing circuit:
                QC = self._convolution_layer(operation, parameters[(param_iter-param_length):param_iter]).compose(QC, range(reversed_input_qubits))
                reversed_input_qubits = operation.reversed_output_qubits
            param_iter -= param_length
        return QC

    def _fully_layer(self,operation: operation, param_vec):
        """
        Builds the quantum circuit of a fully connected layer. Only important if the user don't give the number of qubits (self._num_qubits is None).
        Return:
            returns the quantum circuit.
        """
        gate_params = operation.QC.parameters
        gate_num_qubits = operation.QC.num_qubits
        operation.reversed_input_qubits = gate_num_qubits
        operation.reversed_output_qubits = gate_num_qubits
        QC = QuantumCircuit(gate_num_qubits)
        map_dict = {gate_params[i]:param_vec[i] for i in range(len(gate_params))}
        gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator)) #TODO: Zählung der Operationsnamen: U_0,U_1 usw.
        controlled_qubits = range(gate_num_qubits)
        QC.append(gate, controlled_qubits)
        return QC
       
    def _pooling_layer(self, operation: operation, param_vec):
        """
        Builds the quantum circuit of a pooling layer. Only important if the user don't give the number of qubits (self._num_qubits is None).
        Return:
            returns the quantum circuit.
        """
        gate_params = operation.QC.parameters # params of one block
        gate_num_qubits = operation.QC.num_qubits #TODO: es gibt noch die Attribute output qubits und input qubits, vielleicht kann man diese effizient mit der Generierung der Liste verwenden
        operation.reversed_output_qubits = operation.reversed_input_qubits*gate_num_qubits
        QC = QuantumCircuit(operation.reversed_output_qubits)
        num_gates = operation.reversed_input_qubits
        if operation.var_param:
            map_dict_list = [{gate_params[i]:param_vec[i+k*len(gate_params)] for i in range(len(gate_params))} for k in range(num_gates)]
            qubit_iter = 0
            for map_dict in map_dict_list:
                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                controlled_qubits = range(qubit_iter, gate_num_qubits+qubit_iter)
                QC.append(gate, controlled_qubits)
                qubit_iter += gate_num_qubits
        else:
            map_dict = {gate_params[i]:param_vec[i] for i in range(len(gate_params))}
            qubit_iter = 0
            for i in range(num_gates):
                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                controlled_qubits = range(qubit_iter, gate_num_qubits+qubit_iter)
                QC.append(gate, controlled_qubits)
                qubit_iter += gate_num_qubits
        return QC
    
    def _convolution_layer(self, operation: operation, param_vec):
        """
        Builds the quantum circuit of a convolutional layer. Only important if the user don't give the number of qubits (self._num_qubits is None).
        Return:
            returns the quantum circuit.
        """
        gate_params = operation.QC.parameters
        gate_num_qubits = operation.QC.num_qubits
        operation.reversed_output_qubits = operation.reversed_input_qubits
        QC = QuantumCircuit(operation.reversed_output_qubits)
        if operation.entangled:
            if operation.closed:
                if operation.var_param:
                    num_gates = gate_num_qubits * int(operation.reversed_input_qubits/gate_num_qubits)
                    map_dict_list = [{gate_params[i]:param_vec[i+k*len(gate_params)] for i in range(len(gate_params))} for k in range(num_gates)]
                    map_dict_iter = 0
                    if operation.reversed_input_qubits%gate_num_qubits == 0:
                        for i in range(gate_num_qubits):
                            for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict_list[map_dict_iter],label="{}".format(operation.operator))
                                gate_on_qubits = [t%operation.reversed_input_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                QC.append(gate,gate_on_qubits)
                                map_dict_iter += 1
                    else:
                        gates_to_place_per_block = int(operation.reversed_input_qubits/gate_num_qubits)
                        for i in range(gate_num_qubits):
                            gates_placed = 0 # counts how many gates (vertically) are placed in the ith block
                            enough_gates_placed = False # stores if there are enough gates are placed in the ith iteration, if not, force the first iteration to place as many gates as in the first iteration,
                            #this is important in the case num_qubits < 2*gate_qubits. In that case there must be gates, which control the first and the last qubit simultaneously.
                            for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                                if (operation.reversed_input_qubits-first_gate_qubit>= gate_num_qubits) or not enough_gates_placed:
                                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict_list[map_dict_iter],label="{}".format(operation.operator))
                                    gate_on_qubits = [t%operation.reversed_input_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                    QC.append(gate,gate_on_qubits)
                                    map_dict_iter += 1
                                    gates_placed += 1
                                    if gates_to_place_per_block == gates_placed:
                                        enough_gates_placed = True
                else:
                    map_dict = {gate_params[i]:param_vec[i] for i in range(len(gate_params))}
                    # Testing if num_qubits is dividable by gate_qubits, if so, iterate until every qubit is touched, 
                    # otherwise iterate until there are not enough qubits left for the gate without touching the first qubit (and the last qubit) again:
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                    if operation.reversed_input_qubits%gate_num_qubits == 0:
                        for i in range(gate_num_qubits):
                            for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                                gate_on_qubits = [t%operation.reversed_input_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                QC.append(gate,gate_on_qubits)
                    else:
                        gates_to_place_per_block = int(operation.reversed_input_qubits/gate_num_qubits)
                        for i in range(gate_num_qubits):
                            gates_placed = 0
                            enough_gates_placed = False # stores if there are enough gates are placed in the ith iteration, if not, force the first iteration to place as many gates as in the first iteration, 
                            #this is important in the case num_qubits < 2*gate_qubits. In that case there must be gates, which control the first and the last qubit simultaneously.
                            for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                                if (operation.reversed_input_qubits-first_gate_qubit >= gate_num_qubits) or not enough_gates_placed:
                                    gate_on_qubits = [t%operation.reversed_input_qubits for t in range(first_gate_qubit, first_gate_qubit + gate_num_qubits)]
                                    QC.append(gate,gate_on_qubits)                                    
                                    gates_placed += 1
                                    if gates_to_place_per_block == gates_placed:
                                        enough_gates_placed = True
            else:
                if operation.var_param:
                    num_gates = 0
                    for i in range(gate_num_qubits):
                        for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                            if operation.reversed_input_qubits-first_gate_qubit >= gate_num_qubits:
                                num_gates += 1
                    map_dict_list = [{gate_params[i]:param_vec[i+k*len(gate_params)] for i in range(len(gate_params))} for k in range(num_gates)]
                    map_dict_iter = 0
                    for i in range(gate_num_qubits):
                        for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                            if operation.reversed_input_qubits-first_gate_qubit >= gate_num_qubits:
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict_list[map_dict_iter],label="{}".format(operation.operator))
                                gate_on_qubits = range(first_gate_qubit,first_gate_qubit+gate_num_qubits)
                                QC.append(gate,gate_on_qubits)
                                map_dict_iter += 1
                else:
                    map_dict = {gate_params[i]:param_vec[i] for i in range(len(gate_params))}
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                    for i in range(gate_num_qubits):
                        for first_gate_qubit in range(i, operation.reversed_input_qubits, gate_num_qubits):
                            if operation.reversed_input_qubits-first_gate_qubit >= gate_num_qubits:
                                gate_on_qubits = range(first_gate_qubit,first_gate_qubit+gate_num_qubits)
                                QC.append(gate,gate_on_qubits)
        else:
            num_gates = int(operation.reversed_output_qubits/gate_num_qubits) #TODO: gemeinsamen Namen finden: entweder num_gates oder gate_appearance
            if operation.var_param:
                map_dict_list = [{gate_params[i]:param_vec[i+k*len(gate_params)] for i in range(len(gate_params))} for k in range(num_gates)]
                qubit_iter = 0
                for map_dict in map_dict_list:
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                    gate_on_qubits = range(qubit_iter, qubit_iter+gate_num_qubits)
                    QC.append(gate, gate_on_qubits)
                    qubit_iter += gate_num_qubits
            else:
                map_dict = {gate_params[i]:param_vec[i] for i in range(len(gate_params))}
                qubit_iter = 0
                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                for i in range(num_gates):
                    gate_on_qubits = range(qubit_iter, qubit_iter+gate_num_qubits)
                    QC.append(gate, gate_on_qubits)
                    qubit_iter += gate_num_qubits
        return QC

    
    def get_qubits_left(self):
        """ Returns which qubits the user can control yet. Only possible, if _num_qubits exists (self._num_qubits is None). """
        if self._num_qubits == None:
           raise TypeError("num_qubits doesn't exist, so it's not possible to calculate the qubits left.")
        controlled_qubits = [i for i in range(self.num_qubits)]
        for operation in self.operation_list:
            operation_qubits = operation.QC.num_qubits
            if operation.layer == "p":
                new_controlled_qubits = []
                for i,j in zip(controlled_qubits[0::operation_qubits],controlled_qubits[1::operation_qubits]):
                    new_controlled_qubits.append(j)
                if len(controlled_qubits) %2 != 0:
                    new_controlled_qubits.append(controlled_qubits[-1])
                controlled_qubits = new_controlled_qubits
        return controlled_qubits


    def convolution(self, QC, entangled : bool = False, operator : Union[None,str] = None, var_param : bool = False, target_qubit : Union[int,list,None] = None, closed : bool = False): 
        """
        Adds a convolutional layer to the circuit.
        if entangled is true, it applies a nearest neighbour entangling.
        Args:
            QC: The quantum circuit, which will be applied on every qubit modulo qubits of this circuit, 
                e.g. the QC addresses 2 qubits and the feature map itself has 5 qubits, so the QC will be applied to 1 and 3 (but not 5)
                That means you have to check, if the number of qubits of your QC is correct, otherwise some qubits are not adressed
            entangled: if entangled is true, it applies a nearest neighbour entangling. 
            operator[str,None]: name of the layer  TODO: maybe another name for operator (e.g. label?)
            var_param[bool]: Stores, if the parameters increase by every gate (True) or only per operation (False)
            target_qubit[int,list,None]: Only important, if target_qubit is a list: A single gate is applied on qubits, which are in the target_qubit list. Only possible if self._num_qubits is not None.
            closed[bool]: Only, if entangled is true: Some gates can control first and last qubits and skip qubits in between. If false, gates are only applied on successive qubits.
        """
        if closed and target_qubit != None:
            raise NotImplementedError("It is not implemented to set the target qubit, while there is no num_qubits given. TODO: It might be implemented.") 
        self._add_operation(operation("c",QC,entangled,operator,var_param,target_qubit,closed))

    def pooling(self, QC, operator : Union[None,str] = None, var_param : bool = False, target_qubit: Union[int,None] = None):
        """
        Adds a pooling layer to the circuit.
        Args:
            QC: The quantum circuit, which will be applied on every qubit modulo qubits of this circuit, 
                e.g. the QC addresses 2 qubits and the feature map itself has 5 qubits, so the QC will be applied to 1 and 3 (but not 5)
                That means you have to check, if the number of qubits of your QC is correct, otherwise some qubits are not adressed.
                QC should be an entangling layer, which entangles two or more qubits (for example crx).
            operator[str,None]: name of the layer  TODO: maybe another name for operator (e.g. label?)
            var_param[bool]: Stores, if the parameters increase by every gate (True) or only per operation (False)
            target_qubit[int,None]: Default None: Example with two qubits: it entangles qubit i with qubit i+1 so qubit i gets out of the controlled qubits list in get circuit and i+1 stays. 
                                    If target_qubit is a integer, the qubit with number target_qubit stays and the other qubits go out of the controlled_qubit list (see get_circuit)
        """
        entangled = False
        closed = False
        self._add_operation(operation("p",QC,entangled,operator,var_param,target_qubit,closed))

    def fully_connected(self, QC:QuantumCircuit, operator : Union[None,str] = None):#TODO: bisher muss man ein Gate eingeben, dass genau der Qubit-Größe entspricht, aber der Benutzer wird die genaue Qubitgröße nicht immer wissen
        """
        Adds a fully connected layer to the circuit.
        Args:
            QC: The quantum circuit, which will be applied on every qubit modulo qubits of this circuit, 
                e.g. the QC addresses 2 qubits and the feature map itself has 5 qubits, so the QC will be applied to 1 and 3 (but not 5)
                That means you have to check, if the number of qubits of your QC is correct, otherwise some qubits are not adressed.
                QC must be a gate, which addresses all qubits left
            operator[str,None]: name of the layer  TODO: maybe another name for operator (e.g. label?)
        """
        if self._num_qubits != None:
            if QC.num_qubits > len(self.get_qubits_left()):
                print("Warning on fully connected layer: The quantum circuit input controls too many qubits:",QC.num_qubits,"qubits on input vs.",len(self.get_qubits_left()),"qubits on the actual circuit.")
        var_param = False 
        entangled = False
        target_qubit = None
        closed = False
        self._add_operation(operation("f",QC,entangled,operator,var_param, target_qubit, closed))

