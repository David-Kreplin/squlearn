
import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_gate

from .feature_map_base import FeatureMapBase


class operation():
    def __init__(self, layer: str, QC: QuantumCircuit, entangled: bool, operator, var_param, target_qubit): #TODO: siehe auch convouliton: (...entangled: bool, var_param, operator)
        """
        Args:
            layer[str]: name of the layer (default: "c", "p" or "f")
            QC[QuantumCircuit]: a quantum circuit by qiskit
            operator[str or -1]: Name of the used operator 
            entangled[bool]: only for convolution layers: Stores, if the operation is entangled
           # target[int]: only for pooling layers: Target qubit
            
        """
        self.layer = layer
        self.QC = QC
        self.input_qubits = QC.num_qubits #TODO: überhaupt möglich beim umgekehrten Aufbau ohne dem Wissen über num_qubits?
        if operator == -1:
            if layer == "c":
                self.operator = "U"
                self.output_qubits = self.input_qubits #TODO: input, outputqubits überhaupt nötig?
            elif layer == "p":
                self.operator = "V"
                self.output_qubits = 1
            else:
                self.operator = "F"
                self.output_qubits = self.input_qubits
        else:
            self.operator = operator
            self.output_qubits = self.input_qubits
        self.var_param = var_param
        # not necessary for pooling and fully connected layers:
        self.entangled = entangled
        # only necessary for pooling layers:
        self.target_qubit = target_qubit
        # only necessary for reversed build up of the circuit (if num_qubits is not given)
        # reversed_input_qubits are the qubits on the right side of the gate and reversed_output_qubits the qubits coming out of the left side.
        self.reversed_input_qubits = 0 #has to be overwritten in get_circuit by fully_layer, convolution_layer or pooling_layer
        self.reversed_output_qubits = 0 #has to be overwritten



"""
class convolution(operation):
    def __init__(self, layer: str, QC: QuantumCircuit, entangled: bool, operator, var_param, target_qubit):
        super().__init__(layer, QC, entangled, operator, var_param, target_qubit)
        self.incoming_qubits = [] #TODO
        self.expiring_qubits = self.incoming_qubits
    
class fully_connected(operation):
    def __init__(self, layer: str, QC: QuantumCircuit, entangled: bool, operator, var_param, target_qubit):
        super().__init__(layer, QC, entangled, operator, var_param, target_qubit)
        self.incoming_qubits = []
        self.expiring_qubits = self.incoming_qubits

class pooling(operation):
    def __init__(self, layer: str, QC: QuantumCircuit, entangled: bool, operator, var_param, target_qubit):
        super().__init__(layer, QC, entangled, operator, var_param, target_qubit)
        self.incoming_qubits = [] #TODO                             #example1: list with elements: [1,2,3,4]
        self.expiring_qubits =  self.incoming_qubits[1::2]          #after pooling: [2,4]
        if len(self.incoming_qubits)%2==1:                          #example2: list with elements: [1,2,3,4,5]
            self.expiring_qubits.append(self.incoming_qubits[-1])   #after pooling: [2,4,5]
"""

class qcnn_feature_map(FeatureMapBase):

    def __init__(self, num_qubits: Union[int,None] = None): #TODO: Fehler in Vererbung in num_qubits vermutlich
        self._num_qubits = num_qubits
        self.operation_list = [] # operation list, which contains every used convolution and pooling and also fully_connected operation
        self.parameter_counter = 0  # counts the number of parameters used, if num_qubits is an integer
        #self.param_reversed_counter = 0 # counts the number of parameters used, when number_of _qubits is not given
        self.controlled_qubits = num_qubits # stores, how many qubits can be controlled yet
        self.fully_connected_layer = False


    def get_num_params(self):
        """# TODO: Dies muss später num_params ersetzen
        if anzahl der qubits ist vorgegeben:
            alte Berechnung, ergibt das überhaupt sinn? denn add operation und getnumparams sind völlig verschiedene Methoden
        else:
        """
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
                        if operation.entangled: #TODO: wird später geändert
                            print("muss bei neuem entangled auch geändert werden")
                            gate_appearance = int(num_qubits/gate_num_qubits) + int(num_qubits/gate_num_qubits)
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
                

    def _add_operation(self, operation:operation):
        """adds an operation to the operation_list and increases the parameter counter"""
        # TODO if self.num_qubit == None: -> wird das überhaupt noch benötigt? weil so oder so soll er einen Fehler bei zwei fullys schmeißen?
        if operation.layer == "f":
            if self.fully_connected_layer:
                raise TypeError("There must be at most one fully connected layer.")
            else:
                self.fully_connected_layer = True

        gate_qubits = operation.QC.num_qubits # stores, how many qubits are addressed by the gate #TODO: gate_size vielleicht besserer Name?
        if (self.controlled_qubits != None) and (gate_qubits > self.controlled_qubits):
            if operation.layer == "c":
                print("Warning on convolutional layer: The quantum circuit input controls too many qubits:",gate_qubits,"qubits on input vs.",self.controlled_qubits,"qubits on the actual circuit.")
            elif operation.layer == "p":
                print("Warning on pooling layer: The quantum circuit input controls too many qubits:",gate_qubits,"qubits on input vs.",self.controlled_qubits,"qubits on the actual circuit.")
            else:
                print("Warning on fully connected layer: The quantum circuit input controls too many qubits:",gate_qubits,"qubits on input vs.",self.controlled_qubits,"qubits on the actual circuit.")
        else:
            gate_params = operation.QC.parameters # stores, which parameter vectors are used by the gate
            if operation.var_param:
                if operation.layer == "c":
                    if operation.target_qubit == None:
                        self.parameter_counter += len(gate_params)
                    else:
                        if operation.entangled: 
                            gate_appearance = int(self.controlled_qubits/gate_qubits) + int((self.controlled_qubits-1)/gate_qubits) # stores, how often a gate is applied to the circuit
                        else:
                            gate_appearance = int(self.controlled_qubits/gate_qubits)
                        self.parameter_counter += gate_appearance * len(gate_params)
                elif operation.layer == "p":
                    if self.controlled_qubits != None:
                        qubits_untouched = self.controlled_qubits%gate_qubits
                        gate_appearance = int(self.controlled_qubits/gate_qubits)
                        self.controlled_qubits = gate_appearance + qubits_untouched
                        self.parameter_counter = gate_appearance * len(gate_params)
                else:
                    self.parameter_counter += len(gate_params)
            else:
                if self.controlled_qubits != None: 
                    if operation.layer == "p":
                        qubits_untouched = self.controlled_qubits%gate_qubits
                        gate_appearance = int(self.controlled_qubits/gate_qubits)
                        self.controlled_qubits = gate_appearance + qubits_untouched
                    self.parameter_counter += len(gate_params)
            self.operation_list.append(operation)

    @property
    def num_qubits(self) -> int:
        """ Returns the number of qubits of the feature map. """
        if self.num_qubits == None:
            num_qubits = 1
            for operation in reversed(self.operation_list):
                gate_qubits = operation.QC.num_qubits
                if operation.layer == "f":
                    num_qubits = gate_qubits
                elif operation.layer == "p":
                    num_qubits = gate_qubits*num_qubits
            return num_qubits #TODO: soll auch self.qubits überschrieben werden?
        else:
            return self.num_qubits

    @ property
    def num_features(self) -> int:
        """ Returns the dimension of features of the feature map. """
        return self.num_qubits

    @ property
    def num_parameters(self) -> int:
        """ Returns the number of trainable parameters of the feature map. """
        if self.num_qubits == None:
            param_reversed_counter = 0
            num_qubits = 1
            for operation in reversed(self.operation_list):
                gate_qubits = operation.QC.num_qubits
                gate_params = operation.QC.parameters
                if operation.layer == "f":
                    param_reversed_counter = len(gate_params)
                    num_qubits = gate_qubits
                elif operation.layer == "c":
                    if operation.target_qubit == None:
                        if operation.var_param:
                            if operation.entangled: #TODO: wird später geändert
                                print("muss bei neuem entangled auch geändert werden")
                                gate_appearance = int(num_qubits/gate_qubits) + int(num_qubits/gate_qubits)
                            else:
                                gate_appearance = int(num_qubits/gate_qubits)
                            param_reversed_counter += gate_appearance*len(gate_params)
                        else:
                            param_reversed_counter += len(gate_params)
                    else: # TODO: option einer Liste vielleicht weglassen, da var_param und entangled an der Stelle sowieso keine Bedeutung haben
                        param_reversed_counter == len(gate_params)
                elif operation.layer == "p":
                    num_qubits = gate_qubits*num_qubits
                    gate_appearance = int(num_qubits/gate_qubits)
                    param_reversed_counter += gate_appearance*len(gate_params)
                else:
                    raise NameError("Unknown operation layer.")
            return param_reversed_counter
        else:
            return self.parameter_counter
    
    def get_circuit(self,
                    features: Union[ParameterVector,np.ndarray],    # TODO: features? wie bringen wir die mit rein?
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
        QC = QuantumCircuit(self.num_qubits)
        controlled_qubits = [i for i in range(self.num_qubits)]
        label_name_dict = {"c":0,"p":0} # is needed for counting the name of the operator, if the user doesn't defines it by himself
        global_param_counter = 0 # is required for assigning the local parameters (in every gate) to the global parameter
        for operation in self.operation_list:
            if len(controlled_qubits) == 0:
                raise IndexError("There are to many pooling layers, because there are no qubits left.")
            gate_qubits = operation.QC.num_qubits # with that, the algorithm knows, how many qubits are addressed by the gate
            gate_params = operation.QC.parameters # stores, which parameter vectors are used by the gate
            if operation.layer == "c":
                if operation.target_qubit == None:
                    if operation.entangled:
                        if operation.var_param:
                            gate_appearance = int(len(controlled_qubits)/gate_qubits) + int((len(controlled_qubits)-1)/gate_qubits) # stores, how often the gate appears in this operation (we have got a sum because of the entangling)
                            param_appearance = gate_appearance * len(gate_params) # stores, how many parameters are used in this operation
                            # To store every map for every gate: # [{x1:y1,x2:y2},{x1:y3,x2:y4}]
                            map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(gate_appearance)] # TODO: could be more efficient, if one build gate in for loop without the list
                            global_param_counter += param_appearance
                            map_iter = 0
                            for i,j in zip(range(0,len(controlled_qubits),gate_qubits),range(gate_qubits-1,len(controlled_qubits),gate_qubits)):
                                # the index j is required to end the for loop soon enough. Otherwise it could drive qubits, which don't exist, e.g.: 
                                # The gate drives 3 qubits and the QC has 4 qubits, if i = 2 (so it drives 2,3 and 4) the  for loop must stop immediately after that iteration   
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))                         
                                # the second loop is required to build a list of qubits, which will be controlled applying the gate:
                                gate_on_qubits = []
                                for k in range(i,i+gate_qubits):
                                    gate_on_qubits.append(controlled_qubits[k])
                                QC.append(gate, gate_on_qubits)
                            for i,j in zip(range(1,len(controlled_qubits),gate_qubits),range(gate_qubits,len(controlled_qubits),gate_qubits)):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                # The second loop here is required, to entangle the ith qubit with the (i+1)th. So in this loop the gates will be applied to a qubit, which is offset by one.
                                gate_on_qubits = []
                                for k in range(i,i+gate_qubits):
                                    gate_on_qubits.append(controlled_qubits[k])
                                QC.append(gate, gate_on_qubits)
                            # Attention: With 3 or more qubits it can happen , that the gate layer doesn't reach every qubit, e.g.:
                            # we have a system with 5 qubits and a gate supplies to 3 qubits. Than the gate layer do not reach qubit 3 and 4 (if counting begins at 0). 
                        else:
                            map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                            global_param_counter += len(gate_params)
                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                            for i,j in zip(range(0,len(controlled_qubits),gate_qubits),range(gate_qubits-1,len(controlled_qubits),gate_qubits)):
                                # the index j is required to end the for loop soon enough. Otherwise it could drive qubits, which don't exist, e.g.: 
                                # The gate drives 3 qubits and the QC has 4 qubits, if i = 2 (so it drives 2,3 and 4) the  for loop must stop immediately after that iteration                            
                                # the second loop is required to build a list of qubits, which will be controlled applying the gate:
                                gate_on_qubits = []
                                for k in range(i,i+gate_qubits):
                                    gate_on_qubits.append(controlled_qubits[k])
                                QC.append(gate, gate_on_qubits)
                            for i,j in zip(range(1,len(controlled_qubits),gate_qubits),range(gate_qubits,len(controlled_qubits),gate_qubits)):
                                # The second loop here is required, to entangle the ith qubit with the (i+1)th. So in this loop the gates will be applied to a qubit, which is offset by one.
                                gate_on_qubits = []
                                for k in range(i,i+gate_qubits):
                                    gate_on_qubits.append(controlled_qubits[k])
                                QC.append(gate, gate_on_qubits)
                            # Attention: With 3 or more qubits it can happen , that the gate layer doesn't reach every qubit, e.g.:
                            # we have a system with 5 qubits and a gate supplies to 3 qubits. Than the gate layer do not reach qubit 3 and 4 (if counting begins at 0). 
                    else:
                        if operation.var_param:
                            gate_appearance = int(len(controlled_qubits)/gate_qubits) # stores, how often a gate is applied to the circuit
                            param_appearance = gate_appearance * len(gate_params) # stores, how many parameters are used in this operation
                            # To store every map for every gate:
                            map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(gate_appearance)] # TODO: could be more efficient, if one build gate in for loop without the list
                            global_param_counter += param_appearance
                            map_iter = 0
                            for i,j in zip(range(0,len(controlled_qubits),gate_qubits),range(gate_qubits-1,len(controlled_qubits),gate_qubits)):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                                gate_on_qubits = []
                                for k in range(i,i+gate_qubits):
                                    gate_on_qubits.append(controlled_qubits[k])
                                QC.append(gate, gate_on_qubits)
                        else:
                            map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                            global_param_counter += len(gate_params)
                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                            for i,j in zip(range(0,len(controlled_qubits),gate_qubits),range(gate_qubits-1,len(controlled_qubits),gate_qubits)):
                                gate_on_qubits = []
                                for k in range(i,i+gate_qubits):
                                    gate_on_qubits.append(controlled_qubits[k])
                                QC.append(gate, gate_on_qubits)
                elif type(operation.target_qubit) == list:
                    param_appearance = len(gate_params)
                    map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(param_appearance)}
                    global_param_counter += param_appearance
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                    gate_on_qubits = operation.target_qubit
                    QC.append(gate,gate_on_qubits)
                    
                else: # Target qubit is an integer, so the first controlled qubit will be the target qubit; example: Gate controls 3 qubits, target qubit is 2, so the gate controls qubit 2,3 and 4
                    param_appearance = len(gate_params)
                    map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(param_appearance)}
                    global_param_counter += param_appearance
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["c"]))
                    gate_on_qubits = []
                    for k in range(operation.target_qubit,operation.target_qubit+gate_qubits):
                        gate_on_qubits.append(controlled_qubits[k])
                    QC.append(gate, gate_on_qubits)
                operator_number = label_name_dict["c"]
                label_name_dict.update({"c":operator_number+1})

            elif operation.layer == "p": # Default: last qubit is always the target qubit
                #if gate_qubits != 2: # TODO: Abfrage veraltet, da jetzt auch mehr qubits zulässig sein sollen
                #    raise ValueError("There should be two qubit circuits in a pooling layer.")
                if len(controlled_qubits)<gate_qubits:
                    print("Warning: This pooling operation will not be executed, because there are not enough qubits left.")
                else:
                    if operation.target_qubit == None:
                        if operation.var_param:
                            gate_appearance = int(len(controlled_qubits)/gate_qubits) # stores, how often a gate is applied to the circuit
                            param_appearance = gate_appearance * len(gate_params) # stores, how many parameters are used in this operation
                            # To store every map for every gate:
                            map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(gate_appearance)] # TODO: could be more efficient, if one build gate in for loop without the list
                            global_param_counter += param_appearance
                            map_iter = 0
                            new_controlled_qubits = []
                            """
                            # Only works if there is a gate, which adresses exactly two qubits:
                            for i,j in zip(controlled_qubits[0::gate_qubits],controlled_qubits[1::gate_qubits]):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                                QC.append(gate, [i,j])
                                new_controlled_qubits.append(j)
                            """
                            # last_qubit = 0 # the last qubit, which is controlled by the lowest gate
                            # for i,j in zip(controlled_qubits[0::gate_qubits], controlled_qubits[gate_qubits::gate_qubits]):
                            #     map_dict = map_dict_list[map_iter]
                            #     map_iter += 1
                            #     gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                            #     QC.append(gate, [x for x in range(i,j+1)])
                            #     new_controlled_qubits.append(j)
                            #     last_qubit = j
                            last_qubit = 0
                            for i,j in zip(range(len(controlled_qubits))[0::gate_qubits], controlled_qubits[gate_qubits-1::gate_qubits]):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                                QC.append(gate, controlled_qubits[i:i+gate_qubits])
                                new_controlled_qubits.append(controlled_qubits[i])
                                last_qubit = j
                        else:
                            map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                            global_param_counter += len(gate_params)
                            gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                            new_controlled_qubits = []
                            """
                            # Only works if there is a gate, which adresses exactly two qubits:
                            for i,j in zip(controlled_qubits[0::gate_qubits],controlled_qubits[1::gate_qubits]): # TODO!!!!!!!!!!!!!!!!!!!!!!: noch ändern auf mehr als 2 qubits
                                QC.append(gate, [i,j])
                                new_controlled_qubits.append(j)
                            """
                            # Works with gates of length \geq 2:
                            last_qubit = 0
                            #print("gate_qubits", gate_qubits)
                            #range(len(controlled_qubits))[0::gate_qubits]
                            #for i,j in zip(controlled_qubits[0::gate_qubits], controlled_qubits[gate_qubits-1::gate_qubits]):
                            for i,j in zip(range(len(controlled_qubits))[0::gate_qubits], controlled_qubits[gate_qubits-1::gate_qubits]):
                                #print([x for x in range(i,j+1)])
                                #print("i",i,"j",j)
                                #QC.append(gate, [x for x in range(i,j+1)])
                                QC.append(gate, controlled_qubits[i:i+gate_qubits])
                                new_controlled_qubits.append(j)
                                last_qubit = j
                        """
                        # Checks, if the number of controlled qubits is even or odd
                        # If its odd, add the last qubit to the list (because the last qubit is not affected by controlled operations)
                        if len(controlled_qubits) %2 != 0:
                            new_controlled_qubits.append(controlled_qubits[-1])
                        controlled_qubits = new_controlled_qubits
                        """
                        # checks how many qubits are untouched by pooling gates and adds them to the controlled qubits list
                        for i in range(last_qubit+1, self.num_qubits):
                            new_controlled_qubits.append(i)
                        controlled_qubits = new_controlled_qubits
                        operator_number = label_name_dict["p"]
                        label_name_dict.update({"p":operator_number+1})
                    else: # operation.target_qubit is an integer, so the user chose the remaining target qubit                    
                        if operation.var_param:
                            gate_appearance = int(len(controlled_qubits)/gate_qubits) # stores, how often a gate is applied to the circuit
                            param_appearance = gate_appearance * len(gate_params) # stores, how many parameters are used in this operation
                            # To store every map for every gate:
                            map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(gate_appearance)] # TODO: could be more efficient, if one build gate in for loop without the list
                            global_param_counter += param_appearance
                            map_iter = 0
                            new_controlled_qubits = []
                            """
                            # Only works if there is a gate, which adresses exactly two qubits:
                            for i,j in zip(controlled_qubits[0::gate_qubits],controlled_qubits[1::gate_qubits]):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                                QC.append(gate, [i,j])
                                new_controlled_qubits.append(j)
                            """
                            remaining_qubit = operation.target_qubit # über die if-Schleife setzen
                            last_qubit = 0
                            for i,j in zip(range(len(controlled_qubits))[0::gate_qubits], controlled_qubits[gate_qubits-1::gate_qubits]):
                                map_dict = map_dict_list[map_iter]
                                map_iter += 1
                                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                                QC.append(gate, controlled_qubits[i:i+gate_qubits])
                                #new_controlled_qubits.append(i+remaining_qubit)
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
                            for i,j in zip(range(len(controlled_qubits))[0::gate_qubits], controlled_qubits[gate_qubits-1::gate_qubits]):
                                QC.append(gate, controlled_qubits[i:i+gate_qubits])
                                #new_controlled_qubits.append(i+remaining_qubit)
                                new_controlled_qubits.append(controlled_qubits[i+remaining_qubit])
                                last_qubit = j
                        # checks how many qubits are untouched by pooling gates and adds them to the controlled qubits list
                        for i in range(last_qubit+1, self.num_qubits):
                            new_controlled_qubits.append(i)
                        controlled_qubits = new_controlled_qubits
                        operator_number = label_name_dict["p"]
                        label_name_dict.update({"p":operator_number+1})

            else: # For fully-connected layer
                # TODO: muss noch geändert werden, oder auch nicht?
                if operation.QC.num_qubits != len(controlled_qubits):
                    raise ValueError("Fully-connected gate size is not the same as the number of qubits, that are left. Gate size: "+str(operation.QC.num_qubits)+", number of qubits left: "+str(len(controlled_qubits)))
                map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                param_appearance = len(gate_params)
                global_param_counter += param_appearance
                gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
                QC.append(gate, [k for k in controlled_qubits])
        if global_param_counter < len(parameters):
            raise IndexError("Input parameter vector has a greater size as needed: Input paramvec: ", len(parameters), " size needed:", global_param_counter )
        return QC


    def get_circuit_without_qubits(self,
                                   features: Union[ParameterVector,np.ndarray],
                                   parameters: Union[ParameterVector,np.ndarray]
                                   ) -> QuantumCircuit:
        QC = QuantumCircuit(1)
        param_iter = len(parameters)  #Example: 5 Parameters (0,1,2,3,4); on fully: 3 Parameters (2,3,4); 
        #that means param_iter is 5 and param_length is 3. fully_layer gets a parametervectorlist with entries [2,3,4], after that param_iter is 5-3 = 2.
        reversed_input_qubits = 1
        fully_exists = False
        other_layer_before_fully = False
        for operation in reversed(self.operation_list): #TODO
            param_length = len(operation.QC.parameters)
            if operation.layer == "f":
                fully_exists = True
                if other_layer_before_fully:
                    raise NameError("It's not allowed to put a fully connected layer before an other layer.")
                QC = self.fully_layer(operation, parameters[(param_iter-param_length):param_iter])
                reversed_input_qubits = operation.reversed_output_qubits
            elif operation.layer == "p":
                if not fully_exists:
                    other_layer_before_fully = True
                operation.reversed_input_qubits = reversed_input_qubits
                gate_qubits = operation.QC.num_qubits
                gate_appearance = operation.reversed_input_qubits
                if operation.var_param:
                    param_length = param_length * gate_appearance
                # building up the list of qubits, which will be left
                # Example: there are four 3-qubit gates, so the first one has 2 as target qubit, the second the fifth as target qubit. The target qubit of the third gate is number 6 and the last taret qubit is 9.
                qubits_right_side = []
                aktive_qubit_iterator = -1
                for i in range(int((reversed_input_qubits+1)/2)): # damit rundet das System reversed_input_qubits immer auf
                    aktive_qubit_iterator += gate_qubits
                    qubits_right_side.append(aktive_qubit_iterator)
                aktive_qubit_iterator += 1
                for i in range(int((reversed_input_qubits+1)/2),reversed_input_qubits):
                    qubits_right_side.append(aktive_qubit_iterator)
                    aktive_qubit_iterator += gate_qubits
                QC = self.pooling_layer(operation, parameters[(param_iter-param_length):param_iter]).compose(QC,qubits_right_side)
                reversed_input_qubits = operation.reversed_output_qubits
            else:
                if not fully_exists:
                    other_layer_before_fully = True
                if operation.var_param:
                    pass #TODO:
                    # if operation.entangled:
                    #     if operation.closed:
                    #     else:
                    # else:
                QC = self.convolution_layer(operation, parameters[(param_iter-param_length):param_iter]).compose(QC)
            param_iter -= param_length
        return QC

    def fully_layer(self,operation:operation, param_vec):
        """
        A fully connected layer. Only important if the user don't give the number of qubits.
        """
        gate_params = operation.QC.parameters
        gate_num_qubits = operation.QC.num_qubits
        operation.reversed_input_qubits = gate_num_qubits
        operation.reversed_output_qubits = gate_num_qubits
        QC = QuantumCircuit(gate_num_qubits)
        map_dict = {gate_params[i]:param_vec[i] for i in range(len(gate_params))}
        gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}".format(operation.operator))
        controlled_qubits = range(gate_num_qubits)
        QC.append(gate, controlled_qubits)
        return QC
       
    def pooling_layer(self, operation:operation, param_vec):
        """
        A pooling layer. Only important if the user don't give the number of qubits.
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
    
    def convolution_layer(self, operation, param_vec):
        """
        A convolutional layer. Only important if the user don't give the number of qubits.
        """
        return None #TODO
    
    def get_qubits_left(self):
        """ Returns which qubits the user can control yet. Only possible, if num_qubits exists. """
        if self.num_qubits == None:
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


    def convolution(self, QC, entangled : bool = False, operator = -1, var_param : bool = False, target_qubit : Union[int,list,None] = None): #TODO: operator in label umbenennen; # TODO: -1 in none umbenennen # TODO: target_qubits als liste
        # Zielqubit für gate? also z.b. Zielqubit für ein Gate der größe 3 ist 2, also wird 2,3 und 4 angesteuert, 
        # auf die qubits danach wird dann das Gate auch angewandt (so lange wie es mit der Anzahl der Qubits Sinn ergibt), aber sollen auch Gates davor angewandt werden? (nein oder?)
        # TODO: vielleicht: alle Zielqubits? für 3er Gate: z.B. qubit 1,3 und 4 ansprechen? (oder zu kompliziert und unnötig?)
        """if entangled is true, it applies a nearest neighbour entangling
        Args:
            QC: The quantum circuit, which will be applied on every qubit modulo qubits of this circuit, 
                e.g. the QC addresses 2 qubits and the feature map itself has 5 qubits, so the QC will be applied to 1 and 3 (but not 5)
                That means you have to check, if the number of qubits of your QC is correct, otherwise some qubits are not adressed
            entangled: if true, it applies the QC on every qubit modulo qubits of this circuit beginning at 0 and it applies the QC on every qubit beginning at 1
            operator: name of the layer
        """
        #if QC.num_qubits > len(self.get_qubits_left()): #TODO: veraltet, ist jetzt in addoperation eingebaut
        #    print("Warning on convolutional layer: The quantum circuit input controls too many qubits:",QC.num_qubits,"qubits on input vs.",len(self.get_qubits_left()),"qubits on the actual circuit.")
        self._add_operation(operation("c",QC,entangled,operator,var_param,target_qubit))

    def pooling(self, QC, operator = -1, var_param : bool = False, target_qubit: Union[int,None] = None): #TODO: soll mit mehr als 2 qubits funktionieren, eingabe: welche qubits werden angesteuert, und welcher qubit ist der zielqubit, wie bei convolution mit liste
        # TOOD: Anzahl der Qubits, die angesteuert werden können soll n sein (und nur ein qubit soll danach bleiben) (nicht praxisrelevant) #TODO: pooling nur so wie in paint: also keine Extrawürschte und keine Listen
        """
        QC must be an entangling layer, which entangles two qubits (for example crx).
        Default: it entangles qubit i with qubit i+1 so qubit i gets out of the controlled qubits list in get circuit and i+1 stays
        """
        entangled = False
        self._add_operation(operation("p",QC,entangled,operator,var_param,target_qubit)) # TODO: überlegen, ob man nicht entangled  (False) auf optional lässt

    def fully_connected(self, QC:QuantumCircuit, operator = -1):#TODO: bisher muss man ein Gate eingeben, dass genau der Qubit-Größe entspricht, aber der Benutzer wird die genaue Qubitgröße nicht immer wissen
        """QC must be a gate, which adresses all qubits left"""
        if self._num_qubits != None:
            if QC.num_qubits > len(self.get_qubits_left()):
                print("Warning on fully connected layer: The quantum circuit input controls too many qubits:",QC.num_qubits,"qubits on input vs.",len(self.get_qubits_left()),"qubits on the actual circuit.")
        var_param = False 
        entangled = False
        target_qubit = None
        self._add_operation(operation("f",QC,entangled,operator,var_param, target_qubit))

