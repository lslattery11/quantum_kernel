import numpy as np
import itertools
from typing import Union, Optional, List, Callable
from qiskit.circuit import Parameter,ParameterVector
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
import qiskit.quantum_info as qi
from qiskit.quantum_info.operators.symplectic import Pauli

#methods for generating data with a specific advantage for a kernel.

#generate quantum data from a random vector where the random vector is chosen from a gaussian distribution and 
#fed into a feature map and single qubit pauli expectation values are measured. Can be used for regression data or 
#classification data.
def quantum_pauli_data(
    feature_map ,
    n_points: int,
    seed: int = 0,
    regression: bool = True,
    e: float = 0.0,
    rdm_qubits: List[int] = [0],
    operators: List[str] = ['Z'],
    batch_size: int = 10,
    ):
        assert(len(rdm_qubits)==len(operators)),'number of rdm qubits must be equal to the number of operators.'
        
        #generate random vectors.
        np.random.seed(seed)
        feature_dim_dim=feature_map.feature_dim
        x_vec = np.random.rand(n_points,feature_dim_dim)

        feature_map_params = ParameterVector("par", feature_map.feature_dim)
        #use a statevector quantum simulator. We want accurate non-noisy results.
        quantum_instance = QuantumInstance(AerSimulator(method='statevector', shots=1,device='CPU',blocking_enable=True, blocking_qubits=23))

        parameterized_circuit = construct_circuit(feature_map,feature_map_params)
        parameterized_circuit = quantum_instance.transpile(parameterized_circuit,pass_manager=quantum_instance.unbound_pass_manger)
        #compute batch of statevectors...discard after to minimized max_memory requirements.
        expectation_values=np.zeros((n_points,1))
        for min_idx in range(0, n_points, batch_size):
            max_idx = min(min_idx + batch_size, n_points)
            circuits= [parameterized_circuit.assign_parameters(
                    {feature_map_params: x}
                    )
                    for x in x_vec[min_idx:max_idx]
                ]
            if quantum_instance.bound_pass_manger is not None;
                circuits = quantum_instance.transpile(
                                circuits, pass_manager=quantum_instance.bound_pass_manager
                            )
            #here be computational costs
            results = quantum_instance.execute(circuits,had_transpiled=True)
            #add expectation values for each vector.
            for i,j in itertools.product(range(len(results)),range(len(rdm_qubits)),repeat=1):
                sv=results.get_statevector(i)
                qubit=rdm_qubits[j]
                pauli=Pauli(operators[j])
                exp_val=sv.expectation_value(pauli,qubit) 
                expectation_values[min_idx+i] = exp_val
        
        #should be careful here. currently i am adding gaussian noise...but expectation values from a quantum circuit have
        #a ceiling set by the operator norm. Something to consider but probably not an issue for small noise.
        if regression:
            #add gaussian noise e. Note we add at end not during measurment. Subtle difference here for producing Y=f(X)+e.
            noise=np.random.nomral(scale=e,size=expectation_values.shape)
            expectation_values+=noise
            y_vec=expectation_values
        else:
            raise NotImplementedError('classification data method not implemented yet.')

        return x_vec,y_vec

def construct_circuit(
    feature_map ,
    x: ParameterVector,
    ) -> QuantumCircuit:
        #create circuit for data x
        q = QuantumRegister(feature_map.num_qubits, "q")
        c = ClassicalRegister(feature_map.num_qubits, "c")
        qc = QuantumCircuit(q, c,name='qcx')
        x_dict = dict(zip(feature_map.parameters, x))
        psi = feature_map.assign_parameters(x_dict)
        qc.append(psi.to_instruction(), qc.qubits) 
        
        return qc
