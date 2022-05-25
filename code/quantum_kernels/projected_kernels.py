from itertools import cycle
from matplotlib.contour import QuadContourSet
import numpy as np
from typing import Optional, Union, Sequence, Mapping, List
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.circuit import Parameter,ParameterVector
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

class RDM1ProjectedKernel(QuantumKernel):
    """
    Projected quantum kernel using a single qubit RDM. See Eq. L6 in https://arxiv.org/abs/2011.01938.
    additional arguments for gamma (kernel bandwith) and measure_qubits
    """
    def __init__(
        self,
        feature_map: Optional[QuantumCircuit] = None,
        enforce_psd: bool = True,
        batch_size: int = 900,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        user_parameters: Optional[Union[ParameterVector, Sequence[Parameter]]] = None,
        gamma: float = 1.0,
        measured_qubits: List[int] = [0],
    ) -> None:
        super().__init__(feature_map,enforce_psd,batch_size,quantum_instance,user_parameters)
        self.gamma=gamma
        self._measured_qubits=measured_qubits

    def construct_circuit(self,
        x: ParameterVector,
        y: ParameterVector = None,
        measurement: bool = True,
        is_statevector_sim: bool = False,
    ):
        """
        Overrides parent class function. Parent function returns a single circuit for computing overlap.
        This is a modified version where I will return multiple quantum circuits used in computing the reduced
        density matrix of a single qubit.
        """
        # Use the parent class checks.
        # Ensure all user parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_user_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound user parameters ({unbound_params}).
                All user parameters must be bound to numerical values before constructing
                inner product circuit.
                """
            )

        if len(x) != self._feature_map.num_parameters:
            raise ValueError(
                "x and class feature map incompatible dimensions.\n"
                f"x has {len(x)} dimensions, but feature map has {self._feature_map.num_parameters}."
            )
        #create circuit for data x
        qx = QuantumRegister(self._feature_map.num_qubits, "q")
        cx = ClassicalRegister(self._feature_map.num_qubits, "c")
        qcx = QuantumCircuit(qx, cx,name='qcx')
        x_dict = dict(zip(self._feature_map.parameters, x))
        print(self.feature_map.parameters)
        psi_x = self._feature_map.assign_parameters(x_dict)
        qcx.append(psi_x.to_instruction(), qcx.qubits)
        print(psi_x.parameters)
        print(qcx.parameters)
        #create circuit for data y
        qy = QuantumRegister(self._feature_map.num_qubits, "q")
        cy = ClassicalRegister(self._feature_map.num_qubits, "c")
        qcy = QuantumCircuit(qy, cy,name='qcy')
        if y is None:
            y = x
        y_dict = dict(zip(self._feature_map.parameters, y))
        psi_y = self._feature_map.assign_parameters(y_dict)
        qcy.append(psi_y.to_instruction(), qcy.qubits)
        print(qcy.parameters)
        #add measurements to the appropriate qubits.
        qcxs=[]
        qcys=[]
        if not is_statevector_sim and measurement:
            qcx.barrier(qx)
            qcy.barrier(qy)
            #circuit
            for i in self._measured_qubits:
                new_qcx=qcx.copy(name=qcx.name+str(i))
                new_qcx.measure(i,i)
                qcxs.append(new_qcx)
                new_qcy=qcy.copy(name=qcy.name+str(i))
                new_qcy.measure(i,i)
                qcys.append(new_qcy)
        else:
            raise ValueError(
                "Still need to implement statevector_sim support for construct_circuit"
            )
        return qcxs,qcys


    def _compute_overlap(self, idx, results_x,results_y, is_statevector_sim, measurement_basis) -> float:
        """
        Overrides parent class function. Instead of computing overlap, we wil compute
        the projected kernel element.
        """

        if is_statevector_sim:
            raise ValueError(
               "statevector_sim support for not implemented yet."
            )
        else:
            result = results.get_counts(idx)

            kernel_value = result.get(measurement_basis, 0) / sum(result.values())
        return kernel_value

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        r"""
        Modified version of the parent class function in order to work for projected quantum kernels.
        Construct kernel matrix for given data and feature map

        If y_vec is None, self inner product is calculated.
        If using `statevector_simulator`, only build circuits for :math:`\Psi(x)|0\rangle`,
        then perform inner product classically.

        Args:
            x_vec: 1D or 2D array of datapoints, NxD, where N is the number of datapoints,
                                                            D is the feature dimension
            y_vec: 1D or 2D array of datapoints, MxD, where M is the number of datapoints,
                                                            D is the feature dimension

        Returns:
            2D matrix, NxM

        Raises:
            QiskitMachineLearningError:
                - A quantum instance or backend has not been provided
            ValueError:
                - unbound user parameters in the feature map circuit
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        # Ensure all user parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_user_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound user parameters ({unbound_params}).
                All user parameters must be bound to numerical values before evaluating
                the kernel matrix.
                """
            )

        if self._quantum_instance is None:
            raise QiskitMachineLearningError(
                "A QuantumInstance or Backend must be supplied to evaluate a quantum kernel."
            )
        if isinstance(self._quantum_instance, Backend):
            self._quantum_instance = QuantumInstance(self._quantum_instance)

        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)
        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, len(x_vec)))

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = np.reshape(y_vec, (-1, len(y_vec)))

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        if x_vec.shape[1] != self._feature_map.num_parameters:
            try:
                self._feature_map.num_qubits = x_vec.shape[1]
            except AttributeError:
                raise ValueError(
                    "x_vec and class feature map have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, "
                    f"but feature map has {self._feature_map.num_parameters}."
                ) from AttributeError

        if y_vec is not None and y_vec.shape[1] != self._feature_map.num_parameters:
            raise ValueError(
                "y_vec and class feature map have incompatible dimensions.\n"
                f"y_vec has {y_vec.shape[1]} dimensions, but feature map "
                f"has {self._feature_map.num_parameters}."
            )

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        # initialize kernel matrix
        kernel = np.zeros((x_vec.shape[0], y_vec.shape[0]))

        # set diagonal to 1 if symmetric
        if is_symmetric:
            np.fill_diagonal(kernel, 1)

        # get indices to calculate
        if is_symmetric:
            mus, nus = np.triu_indices(x_vec.shape[0], k=1)  # remove diagonal
        else:
            mus, nus = np.indices((x_vec.shape[0], y_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        is_statevector_sim = self._quantum_instance.is_statevector
        measurement = not is_statevector_sim

        # calculate kernel
        if is_statevector_sim:  # using state vector simulator
            raise ValueError(
               "statevector_sim support for not implemented yet."
            )

        else:  # not using state vector simulator
            feature_map_params_x = ParameterVector("par_x", self._feature_map.num_parameters)
            feature_map_params_y = ParameterVector("par_y", self._feature_map.num_parameters)
            parameterized_circuits_x,parameterized_circuits_y = self.construct_circuit(
                feature_map_params_x,
                feature_map_params_y,
                measurement=measurement,
                is_statevector_sim=is_statevector_sim,
            )
            print(len(parameterized_circuits_y))
            print(type(parameterized_circuits_y[0]))
            parameterized_circuits_x = self._quantum_instance.transpile(
                parameterized_circuits_x, pass_manager=self._quantum_instance.unbound_pass_manager
            )
            parameterized_circuits_y = self._quantum_instance.transpile(
                parameterized_circuits_y, pass_manager=self._quantum_instance.unbound_pass_manager
            )
            print(type(parameterized_circuits_y[0]))
            for idx in range(0, len(mus), self._batch_size):
                to_be_computed_data_pair = []
                to_be_computed_index = []
                for sub_idx in range(idx, min(idx + self._batch_size, len(mus))):
                    i = mus[sub_idx]
                    j = nus[sub_idx]
                    x_i = x_vec[i]
                    y_j = y_vec[j]
                    if not np.all(x_i == y_j):
                        to_be_computed_data_pair.append((x_i, y_j))
                        to_be_computed_index.append((i, j))
                #need to change here.
                print('here')
                circuits_x = [[
                    parameterized_circuits_x[idx].assign_parameters(
                        {feature_map_params_x: x}
                    )
                    for idx in range(len(parameterized_circuits_x))
                ]
                    for x, y in to_be_computed_data_pair
                ]
                print('here2')
                circuits_y = [[
                    parameterized_circuits_y[idx].assign_parameters(
                        {feature_map_params_y: y}
                    )
                    for idx in range(len(parameterized_circuits_y))
                ]
                    for x, y in to_be_computed_data_pair
                ]
                print('here3')
                print(circuits_y[0])

                if self._quantum_instance.bound_pass_manager is not None:
                    circuits_x = self._quantum_instance.transpile(
                        circuits_x, pass_manager=self._quantum_instance.bound_pass_manager
                    )
                    circuits_y = self._quantum_instance.transpile(
                        circuits_y, pass_manager=self._quantum_instance.bound_pass_manager
                    )
                print(circuits_y[0])
                results_x = self._quantum_instance.execute(circuits_x, had_transpiled=True)
                results_y = self._quantum_instance.execute(circuits_y, had_transpiled=True)
                #setting to None, can change if I it turns out I want to use a specific basis and deviate
                #from the paper.
                measurement_basis = None
                return results_x,results_y
            """                
                matrix_elements = [

                    self._compute_overlap(idx, results_x,results_y, is_statevector_sim, measurement_basis)
                    for idx in range(len(circuits_x))

                ]

                for (i, j), value in zip(to_be_computed_index, matrix_elements):
                    kernel[i, j] = value
                    if is_symmetric:
                        kernel[j, i] = kernel[i, j]

            if self._enforce_psd and is_symmetric:
                # Find the closest positive semi-definite approximation to symmetric kernel matrix.
                # The (symmetric) matrix should always be positive semi-definite by construction,
                # but this can be violated in case of noise, such as sampling noise, thus the
                # adjustment is only done if NOT using the statevector simulation.
                D, U = np.linalg.eig(kernel)  # pylint: disable=invalid-name
                kernel = U @ np.diag(np.maximum(0, D)) @ U.transpose()
 
        return kernel
            """