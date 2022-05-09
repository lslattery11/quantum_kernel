import numpy as np
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit import Parameter,ParameterVector

class Heisenberg1DFeatureMap(BlueprintCircuit):
    """
    Feature map for time evolution of the 1D Heisenberg model. H = . Time evolution is given by first order Suzuki-Trotter.
    Currently implemented with OBC as in Ruslan's original paper but I think PBC makes more sense for long-range interaction
    so that each feature can be equally weighted in the Hamiltonian.
    """
    def __init__(
        self,
        feature_dim: int,
        n_trotter: int,
        evo_time: float,
        init_state: str,
        init_state_seed: int,
        k : int = 1,
        r : float = 0.0,
        name: str = 'Heisenberg1D'
    ) -> None:
        super().__init__(name=name)
        self._num_qubits = feature_dim+1
        self._feature_dimension = feature_dim
        self._support_parameterized_circuit = True
        self._init_state=init_state
        self._init_state_seed=init_state_seed

        self.scaling_factor= 2 * evo_time/n_trotter
        self.k=k
        self.r=r
        self.evo_time=evo_time
        self.n_trotter=n_trotter

        self.add_register(QuantumRegister(self._num_qubits,'q'))

    def _build(self):
        super()._build()
        qc=self.construct_circuit(self.qregs)
        inst=qc.to_instruction()
        qargs=qc.qubits
        cargs=qc.clbits
        self._data=[(inst,qargs,cargs)]
        self._update_parameter_table(inst)
        return

    def _check_configuration(self) -> bool:
        valid=False
        if self._num_qubits>=self._feature_dimension:
            valid=True
        return valid

    def construct_circuit(self,qr=None,inverse=False):
        params = ParameterVector(name='x', length=self._feature_dimension)
        qr=self.qregs[0]
        qc = QuantumCircuit(qr)

        qc=self.construct_init_state(qc,qr)
        np.random.seed()
        for _ in range(self.n_trotter):
            for l in range(1,self.k+1):
                for q1 in range(self._feature_dimension):
                    q2=q1+l
                    if q2 >= self._num_qubits:
                        continue
                    #decay of long range interactions
                    decay=1/(l**self.r)
                    # XX
                    qc.h([qr[q1],qr[q2]])
                    qc.cx(qr[q1],qr[q2])
                    qc.rz(decay*params[q1], qr[q2])
                    qc.cx(qr[q1],qr[q2])
                    qc.h([qr[q1],qr[q2]])
                    # YY
                    qc.rx(np.pi/2, [qr[q1],qr[q2]])
                    qc.cx(qr[q1],qr[q2])
                    qc.rz(decay*params[q1], qr[q2])
                    qc.cx(qr[q1],qr[q2])
                    qc.rx(np.pi/2, [qr[q1],qr[q2]])
                    # ZZ
                    qc.cx(qr[q1],qr[q2])
                    qc.rz(decay*params[q1], qr[q2])
                    qc.cx(qr[q1],qr[q2])
        return qc


    def construct_init_state(self,qc,qr):
        np.random.seed(self._init_state_seed)
        init_state=self._init_state
        if init_state == 'Haar_random':
            # pick and fix Haar-random single-qubit state
            random_params = np.random.uniform(size=(len(qr), 3))
            for i,qreg in enumerate(qr):
                qc.rx(2 * random_params[i][0], qreg)
                qc.ry(2 * random_params[i][1], qreg)
                qc.rz(2 * random_params[i][2], qreg)
        elif init_state == 'Basis_random':
            # random computational basis state
            for i,qreg in enumerate(qr):
                if np.random.randint(0, 2):
                    qc.x(qreg)
        elif init_state == 'X_random':
            # random X rotation on each qubit basis state
            random_params = np.random.uniform(size=(self._feature_dimension))
            for i,qreg in enumerate(qr):
                if np.random.randint(0, 2):
                    qc.rx(random_params[i], qreg)
        else:
            raise ValueError(f"Unknown initial state init_state={init_state}")
        return qc



    