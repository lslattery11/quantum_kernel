import numpy as np
import itertools
import random
from typing import Union, Optional, List, Callable
from qiskit.circuit.library.data_preparation.zz_feature_map import ZZFeatureMap

class Sparse_IQP(ZZFeatureMap):
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        connectivity: str = "full",
        density: Union[str,int] = 'max',
        int_time_scale: float = 1.0,
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "SparseIQPFeatureMap",
    ) -> None:
        """class for an IQP circuit as a feature map with controllable sparsity via the entanglement.

        Args:
            feature_dimension: Number of features.
            reps: The number of repeated circuits, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.NLocal` for detail.
            data_map_func: A mapping function for data x.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        """
        entanglement=get_entanglement(connectivity,density,feature_dimension)

        super().__init__(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        self.int_time_scale=int_time_scale

    def get_entangler_map(self, rep_num: int, block_num: int, num_block_qubits: int
    ):
        """
        add functionality to parent (qiskit's NLocal class) get_entangler_map class method. Would like it for
        the returned entangler map to be the new sparse connections specified by self.entanglement

        """
        
        entanglement=self._entanglement
        if isinstance(entanglement, List):
            return entanglement[num_block_qubits-1]
        else:
            return super().get_entangler_map(rep_num,block_num,num_block_qubits)

    def pauli_evolution(self,pauli_string,time):
        if len(pauli_string) == 2:
            time=self.int_time_scale*time
        return super().pauli_evolution(pauli_string,time)
    
def get_entanglement(connectivity,density,num_qubits):
    """ 
    Create get_entanglement for the Sparse_IQP class's modified NLocal class method get_entangler_map.
    
    NLocal circuits allow multiple forms of entanglements. For our purposes we will restrict ourselves to
    full connectivity and a k-nearest neighbor connectivity. Further, k-local and full connectivity 
    can then have a density setting controlling the number of connections for each qubit in the two qubit interaction term
    for the IQP circuit.
 

    """
    #density > possible combinations should return the same effective entanglement as 'max'.
    #handle full connectivity
    if connectivity=='full':
        if density=='max':
            entanglement=connectivity
        else:
            density=min(density,num_qubits-1)
            entanglement=[]
            #0th index of entanglement is just single # tuples for each qubit.
            qubits=[(qubit,) for qubit in range(0,num_qubits)]
            entanglement.append(qubits)

            #all possible unique qubit pairs
            qubit_pairs=list(itertools.combinations(range(0,num_qubits),2))
            random.shuffle(qubit_pairs)
            #keep pairs up to density number of connections.
            sparse_qubit_pairs=[]
            for pair in qubit_pairs:
                if np.sum([1 for i in sparse_qubit_pairs if pair[0] in i]) < density and np.sum([1 for i in sparse_qubit_pairs if pair[1] in i])< density:
                    sparse_qubit_pairs.append(pair)
            entanglement.append(sparse_qubit_pairs)

    #handle k-local connectivity
    elif connectivity=='klocal':
        entanglement=[]
        #0th index of entanglement is just single # tuples for each qubit.
        qubits=[(qubit,) for qubit in range(0,num_qubits)]
        entanglement.append(qubits)
        #all possible qubit pairs within k=connectivity of each other, currently using obc. (Need to switch)
        qubit_pairs=itertools.combinations(range(0,num_qubits),2)
        klocal_qubit_pairs=[pair for pair in qubit_pairs if np.abs(pair[0]-pair[1]) <= density]
        entanglement.append(klocal_qubit_pairs)
        #else each qubit should be connected 'density' number of times.

    else:
        raise ValueError("connectivity should be 'full' or 'klocal'")

    return entanglement