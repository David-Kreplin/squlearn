
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
        if operator == -1:
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
        # only necessary for pooling layers:
        self.target_qubit = target_qubit
        self.incoming_qubits = []
        self.expiring_qubits = []


class qcnn_feature_map(FeatureMapBase):

    def __init__(self, number_of_qubits):
        self.number_of_qubits = number_of_qubits
        self.operation_list = [] # operation list, which contains every used convolution and pooling and also fully_connected operation
        self.parameter_counter = 0  # counts the number of parameters used
        self.controlled_qubits = number_of_qubits # stores, how many qubits can be controlled yet

    def _add_operation(self, operation:operation):
        """adds an operation to the operation_list and increases the parameter counter"""
        if operation.target_qubit == None:
            gate_params = operation.QC.parameters # stores, which parameter vectors are used by the gate
            if operation.var_param: #TODO: überprüfen, ob bei zu großer Größe des Gates trotzdem die Parameter hochgezählt werden
                gate_qubits = operation.QC.num_qubits # stores, how many qubits are addressed by the gate
                if operation.layer == "c":
                    if operation.entangled: # TODO: controlled qubits statt number-of_qubits
                        gate_appearance = int(self.controlled_qubits/gate_qubits) + int((self.controlled_qubits-1)/gate_qubits) # stores, how often a gate is applied to the circuit
                    else:
                        gate_appearance = int(self.controlled_qubits/gate_qubits)
                    self.parameter_counter += gate_appearance * len(gate_params)
                elif operation.layer == "p":
                    gate_appearance = int(self.controlled_qubits/gate_qubits)
                    self.parameter_counter += gate_appearance * len(gate_params)
                    self.controlled_qubits = int((self.controlled_qubits+1)/2) # the number of controlled qubits after one pooling is int((n+1)/2); Examples: 8 qubits, after pooling: 4; 7 qubits, after pooling: 4
                else:
                    self.parameter_counter += len(gate_params)
            else:
                if operation.layer == "p":
                    self.controlled_qubits = int((self.controlled_qubits+1)/2) # the number of controlled qubits after one pooling is int((n+1)/2); Examples: 8 qubits, after pooling: 4; 7 qubits, after pooling: 4
                self.parameter_counter += len(gate_params)
            
        else:
            gate_params = operation.QC.parameters
            if operation.layer == "c":
                self.parameter_counter += len(gate_params)
            else:
                pass #TODO: vorsicht target param könnte sich in pooling je nachdem stark von convolution unterscheiden
        self.operation_list.append(operation)

    @property
    def num_qubits(self) -> int:
        """ Returns the number of qubits of the feature map. """
        return self.number_of_qubits

    @ property
    def num_features(self) -> int:
        """ Returns the dimension of features of the feature map. """
        return self.number_of_qubits

    @ property
    def num_parameters(self) -> int:
        """ Returns the number of trainable parameters of the feature map. """
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
        QC = QuantumCircuit(self.number_of_qubits)
        controlled_qubits = [i for i in range(self.number_of_qubits)]
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
                else:
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

            elif operation.layer == "p":
                #if gate_qubits != 2: # TODO: Abfrage veraltet, da jetzt auch mehr qubit zulässig sein sollen
                #    raise ValueError("There should be two qubit circuits in a pooling layer.")
                #if operation.target_qubit == None:
                if operation.var_param:
                    gate_appearance = int(len(controlled_qubits)/gate_qubits) # stores, how often a gate is applied to the circuit
                    param_appearance = gate_appearance * len(gate_params) # stores, how many parameters are used in this operation
                    # To store every map for every gate:
                    map_dict_list = [{gate_params[i]:parameters[j+global_param_counter] for i,j in zip(range(len(gate_params)),range(k*len(gate_params),(k+1)*len(gate_params)))} for k in range(gate_appearance)] # TODO: could be more efficient, if one build gate in for loop without the list
                    global_param_counter += param_appearance
                    map_iter = 0
                    new_controlled_qubits = []
                    # Only works if there is a gate, which adresses exactly two qubits:
                    for i,j in zip(controlled_qubits[0::gate_qubits],controlled_qubits[1::gate_qubits]):
                        map_dict = map_dict_list[map_iter]
                        map_iter += 1
                        gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                        QC.append(gate, [i,j])
                        new_controlled_qubits.append(j)
                else:
                    map_dict = {gate_params[i]:parameters[i+global_param_counter] for i in range(len(gate_params))}
                    global_param_counter += len(gate_params)
                    gate = circuit_to_gate(operation.QC, parameter_map=map_dict,label="{}_{}".format(operation.operator,label_name_dict["p"]))
                    new_controlled_qubits = []
                    # Only works if there is a gate, which adresses exactly two qubits:
                    for i,j in zip(controlled_qubits[0::gate_qubits],controlled_qubits[1::gate_qubits]):
                        QC.append(gate, [i,j])
                        new_controlled_qubits.append(j)
                # Checks, if the number of controlled qubits is even or odd
                # If its odd, add the last qubit to the list (because the last qubit is not affected by controlled operations)
                if len(controlled_qubits) %2 != 0:
                    new_controlled_qubits.append(controlled_qubits[-1])
                controlled_qubits = new_controlled_qubits
                operator_number = label_name_dict["p"]
                label_name_dict.update({"p":operator_number+1})
                #elif type(operation.target_qubit) == list:

            else:
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


    def get_qubits_left(self):
        """ Returns which qubits the user can control yet. """
        controlled_qubits = [i for i in range(self.number_of_qubits)]
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
        if QC.num_qubits > len(self.get_qubits_left()):
            print("Warning on convolutional layer: The quantum circuit input controls too many qubits:",QC.num_qubits,"qubits on input vs.",len(self.get_qubits_left()),"qubits on the actual circuit.")
        self._add_operation(operation("c",QC,entangled,operator,var_param,target_qubit))

    def pooling(self, QC, operator = -1, var_param : bool = False, target_qubit: Union[int,list,None] = None): #TODO: soll mit mehr als 2 qubits funktionieren, eingabe: welche qubits werden angesteuert, und welcher qubit ist der zielqubit, wie bei convolution mit liste
        # TOOD: Anzahl der Qubits, die angesteuert werden können soll n sein (und nur ein qubit soll danach bleiben) (nicht praxisrelevant)
        """
        QC must be an entangling layer, which entangles two qubits (for example crx).
        Default: it entangles qubit i with qubit i+1 so qubit i gets out of the controlled qubits list in get circuit and i+1 stays
        """
        entangled = False
        self._add_operation(operation("p",QC,entangled,operator,var_param,target_qubit)) # TODO: überlegen, ob man nicht entangled  (False) auf optional lässt

    def fully_connected(self, QC, operator = -1):#TODO: bisher muss man ein Gate eingeben, dass genau der Qubit-Größe entspricht, aber der Benutzer wird die genaue Qubitgröße nicht immer wissen
        """QC must be a gate, which adresses all qubits left"""
        if QC.num_qubits > len(self.get_qubits_left()):
            print("Warning on fully connected layer: The quantum circuit input controls too many qubits:",QC.num_qubits,"qubits on input vs.",len(self.get_qubits_left()),"qubits on the actual circuit.")
        var_param = False 
        entangled = False
        target_qubit = None
        self._add_operation(operation("f",QC,entangled,operator,var_param, target_qubit))

