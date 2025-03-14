from abc import ABC, abstractmethod
import numpy as np
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import scipy.linalg as la

class ErrorCorrectionCode(Enum):
    """Enumeration of supported quantum error correction codes."""
    NONE = auto()
    REPETITION = auto()
    STEANE = auto()
    SHOR = auto()
    SURFACE = auto()
    STABILIZER = auto()

class QuantumState(ABC):
    """
    Abstract base class for quantum states.
    
    Provides a common interface for different representations of quantum states,
    including pure states (StateVector) and mixed states (DensityMatrix).
    """
    
    @abstractmethod
    def __init__(self, dimensions: int):
        """
        Initialize a quantum state with specified dimensions.
        
        Args:
            dimensions: Number of dimensions in the Hilbert space
        """
        self.dimensions = dimensions
        self.error_correction = ErrorCorrectionCode.NONE
    
    @abstractmethod
    def evolve(self, operator: np.ndarray) -> None:
        """
        Evolve the quantum state under a given operator.
        
        Args:
            operator: The operator to evolve the state under
        """
        pass
    
    @abstractmethod
    def measure(self, observable: np.ndarray) -> Tuple[float, 'QuantumState']:
        """
        Measure the quantum state with respect to an observable.
        
        Args:
            observable: The observable to measure
            
        Returns:
            Tuple containing the measurement result and the post-measurement state
        """
        pass
    
    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """
        Calculate the probability distribution of the quantum state.
        
        Returns:
            Array of probabilities for each basis state
        """
        pass
    
    @abstractmethod
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate the fidelity between this quantum state and another.
        
        Args:
            other: The quantum state to compare with
            
        Returns:
            Fidelity value between 0 and 1
        """
        pass
    
    @abstractmethod
    def to_density_matrix(self) -> 'DensityMatrix':
        """
        Convert the quantum state to a density matrix representation.
        
        Returns:
            DensityMatrix representation of the state
        """
        pass
    
    def enable_error_correction(self, code: ErrorCorrectionCode) -> None:
        """
        Enable error correction for this quantum state.
        
        Args:
            code: The error correction code to use
        """
        self.error_correction = code
    
    @abstractmethod
    def apply_error_correction(self) -> None:
        """Apply the configured error correction to the quantum state."""
        pass
    
    @classmethod
    @abstractmethod
    def from_bloch(cls, theta: float, phi: float) -> 'QuantumState':
        """
        Create a quantum state from Bloch sphere coordinates.
        
        Args:
            theta: Polar angle (0 to pi)
            phi: Azimuthal angle (0 to 2pi)
            
        Returns:
            Quantum state representing the specified point on the Bloch sphere
        """
        pass


class StateVector(QuantumState):
    """
    Representation of a pure quantum state as a state vector.
    
    A pure quantum state is represented by a complex vector in Hilbert space.
    """
    
    def __init__(self, vector: Optional[np.ndarray] = None, dimensions: int = 2):
        """
        Initialize a state vector.
        
        Args:
            vector: Initial state vector. If None, defaults to |0⟩
            dimensions: Hilbert space dimensions
        """
        super().__init__(dimensions)
        
        if vector is None:
            # Default to |0⟩ state
            self.vector = np.zeros(dimensions, dtype=complex)
            self.vector[0] = 1.0
        else:
            if len(vector) != dimensions:
                raise ValueError(f"Vector dimension {len(vector)} does not match specified dimensions {dimensions}")
            
            # Normalize the vector
            norm = np.linalg.norm(vector)
            if norm < 1e-10:
                raise ValueError("State vector has zero norm")
            
            self.vector = vector / norm
    
    def evolve(self, operator: np.ndarray) -> None:
        """
        Evolve the state vector under a unitary operator.
        
        Args:
            operator: Unitary operator for evolution
        """
        if operator.shape != (self.dimensions, self.dimensions):
            raise ValueError(f"Operator dimensions {operator.shape} do not match state dimensions {self.dimensions}")
        
        self.vector = operator @ self.vector
        
        # Renormalize to account for numerical errors
        norm = np.linalg.norm(self.vector)
        if norm > 1e-10:
            self.vector /= norm
    
    def measure(self, observable: np.ndarray) -> Tuple[float, 'StateVector']:
        """
        Measure the state with respect to an observable.
        
        Args:
            observable: Hermitian operator representing the observable
            
        Returns:
            Tuple containing the measurement result and post-measurement state
        """
        if observable.shape != (self.dimensions, self.dimensions):
            raise ValueError(f"Observable dimensions {observable.shape} do not match state dimensions {self.dimensions}")
        
        # Verify observable is Hermitian
        if not np.allclose(observable, observable.conj().T):
            raise ValueError("Observable must be Hermitian")
        
        # Calculate expectation value
        expectation = np.real(self.vector.conj() @ observable @ self.vector)
        
        # For a projective measurement, we would collapse to an eigenstate
        # This is a simplified implementation
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Calculate probabilities for each eigenstate
        probs = np.zeros_like(eigenvalues, dtype=float)
        for i, eigenvector in enumerate(eigenvectors.T):
            probs[i] = np.abs(np.vdot(eigenvector, self.vector))**2
        
        # Select an outcome based on probabilities
        outcome_idx = np.random.choice(self.dimensions, p=probs)
        outcome = eigenvalues[outcome_idx]
        
        # Collapse to the corresponding eigenstate
        new_state = StateVector(eigenvectors[:, outcome_idx], self.dimensions)
        
        return outcome, new_state
    
    def probabilities(self) -> np.ndarray:
        """
        Calculate measurement probabilities in the computational basis.
        
        Returns:
            Array of probabilities for each basis state
        """
        return np.abs(self.vector)**2
    
    def fidelity(self, other: QuantumState) -> float:
        """
        Calculate the fidelity with another quantum state.
        
        For pure states, this is |⟨ψ|φ⟩|²
        
        Args:
            other: The quantum state to compare with
            
        Returns:
            Fidelity value between 0 and 1
        """
        if isinstance(other, StateVector):
            return np.abs(np.vdot(self.vector, other.vector))**2
        else:
            # If comparing with a mixed state, use the mixed state's implementation
            return other.fidelity(self)
    
    def to_density_matrix(self) -> 'DensityMatrix':
        """
        Convert to density matrix representation ρ = |ψ⟩⟨ψ|.
        
        Returns:
            DensityMatrix representation of this state
        """
        # Calculate outer product |ψ⟩⟨ψ|
        density_matrix = np.outer(self.vector, self.vector.conj())
        return DensityMatrix(density_matrix)
    
    def apply_error_correction(self) -> None:
        """Apply configured error correction to the state vector."""
        if self.error_correction == ErrorCorrectionCode.NONE:
            return
        
        # Implementation would depend on the specific error correction code
        # This is a placeholder that would be replaced with actual implementation
        if self.error_correction == ErrorCorrectionCode.REPETITION:
            # Apply repetition code error correction
            pass
        elif self.error_correction == ErrorCorrectionCode.STEANE:
            # Apply Steane code error correction
            pass
        elif self.error_correction == ErrorCorrectionCode.SHOR:
            # Apply Shor code error correction
            pass
        elif self.error_correction == ErrorCorrectionCode.SURFACE:
            # Apply surface code error correction
            pass
        elif self.error_correction == ErrorCorrectionCode.STABILIZER:
            # Apply stabilizer code error correction
            pass
    
    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> 'StateVector':
        """
        Create a qubit state from Bloch sphere coordinates.
        
        Args:
            theta: Polar angle (0 to pi)
            phi: Azimuthal angle (0 to 2pi)
            
        Returns:
            StateVector representing the specified point on the Bloch sphere
        """
        # Bloch sphere representation for a single qubit
        vector = np.array([
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2)
        ], dtype=complex)
        
        return cls(vector, dimensions=2)
    
    @classmethod
    def create_bell_state(cls, bell_type: int = 0) -> 'StateVector':
        """
        Create one of the four Bell states.
        
        Args:
            bell_type: Integer from 0-3 specifying which Bell state to create
                0: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                1: |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                2: |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                3: |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        
        Returns:
            StateVector representing the specified Bell state
        """
        if bell_type == 0:  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            vector = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        elif bell_type == 1:  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            vector = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
        elif bell_type == 2:  # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            vector = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
        elif bell_type == 3:  # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            vector = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
        else:
            raise ValueError("bell_type must be an integer from 0-3")
        
        return cls(vector, dimensions=4)


class DensityMatrix(QuantumState):
    """
    Representation of a general (potentially mixed) quantum state as a density matrix.
    
    A density matrix ρ is a positive semidefinite Hermitian matrix with trace 1.
    """
    
    def __init__(self, matrix: Optional[np.ndarray] = None, dimensions: int = 2):
        """
        Initialize a density matrix.
        
        Args:
            matrix: Initial density matrix. If None, defaults to |0⟩⟨0|
            dimensions: Hilbert space dimensions
        """
        super().__init__(dimensions)
        
        if matrix is None:
            # Default to |0⟩⟨0| state
            self.matrix = np.zeros((dimensions, dimensions), dtype=complex)
            self.matrix[0, 0] = 1.0
        else:
            if matrix.shape != (dimensions, dimensions):
                raise ValueError(f"Matrix dimensions {matrix.shape} do not match specified dimensions {dimensions}")
            
            # Ensure the matrix is Hermitian
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Density matrix must be Hermitian")
            
            # Ensure the matrix has trace 1
            trace = np.trace(matrix)
            if not np.isclose(trace, 1.0):
                # Normalize the matrix
                if trace < 1e-10:
                    raise ValueError("Density matrix has zero trace")
                matrix = matrix / trace
            
            # Ensure the matrix is positive semidefinite
            eigenvalues = np.linalg.eigvalsh(matrix)
            if np.any(eigenvalues < -1e-10):
                raise ValueError("Density matrix must be positive semidefinite")
            
            self.matrix = matrix
    
    def evolve(self, operator: np.ndarray) -> None:
        """
        Evolve the density matrix under a unitary operator: ρ → UρU†.
        
        Args:
            operator: Unitary operator for evolution
        """
        if operator.shape != (self.dimensions, self.dimensions):
            raise ValueError(f"Operator dimensions {operator.shape} do not match state dimensions {self.dimensions}")
        
        # ρ → UρU†
        self.matrix = operator @ self.matrix @ operator.conj().T
        
        # Ensure the result is Hermitian (handle numerical errors)
        self.matrix = (self.matrix + self.matrix.conj().T) / 2
        
        # Renormalize to account for numerical errors
        trace = np.trace(self.matrix)
        if trace > 1e-10:
            self.matrix /= trace
    
    def measure(self, observable: np.ndarray) -> Tuple[float, 'DensityMatrix']:
        """
        Measure the state with respect to an observable.
        
        Args:
            observable: Hermitian operator representing the observable
            
        Returns:
            Tuple containing the measurement result and post-measurement state
        """
        if observable.shape != (self.dimensions, self.dimensions):
            raise ValueError(f"Observable dimensions {observable.shape} do not match state dimensions {self.dimensions}")
        
        # Verify observable is Hermitian
        if not np.allclose(observable, observable.conj().T):
            raise ValueError("Observable must be Hermitian")
        
        # Diagonalize the observable
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Calculate probabilities for each eigenstate
        probs = np.zeros_like(eigenvalues, dtype=float)
        projectors = []
        
        for i, eigval in enumerate(eigenvalues):
            # Construct projector onto eigenspace
            eigvec = eigenvectors[:, i].reshape(-1, 1)
            projector = eigvec @ eigvec.conj().T
            projectors.append(projector)
            
            # Calculate probability: p_i

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum, auto


class ErrorCorrectionType(Enum):
    """Enumeration of supported quantum error correction codes."""
    NONE = auto()
    REPETITION = auto()
    STEANE = auto()
    SHOR = auto()
    SURFACE = auto()
    STABILIZER = auto()


class QuantumState(ABC):
    """
    Abstract base class for quantum state representations.
    
    This class defines the interface for all quantum state implementations,
    including pure states (state vectors) and mixed states (density matrices).
    """
    
    @abstractmethod
    def get_num_qubits(self) -> int:
        """Return the number of qubits in the state."""
        pass
    
    @abstractmethod
    def get_state_representation(self):
        """Return the underlying mathematical representation of the state."""
        pass
    
    @abstractmethod
    def apply_gate(self, gate: np.ndarray, target_qubits: List[int]) -> None:
        """
        Apply a quantum gate to the state.
        
        Args:
            gate: Unitary matrix representing the quantum gate
            target_qubits: List of qubit indices the gate acts on
        """
        pass
    
    @abstractmethod
    def measure(self, qubit_indices: List[int] = None) -> Tuple[int, 'QuantumState']:
        """
        Perform a measurement on the specified qubits.
        
        Args:
            qubit_indices: Indices of qubits to measure. If None, measure all qubits.
            
        Returns:
            Tuple of (measurement outcome, post-measurement state)
        """
        pass
    
    @abstractmethod
    def get_probability(self, state_index: int) -> float:
        """
        Get the probability of measuring a specific state.
        
        Args:
            state_index: Index of the computational basis state
            
        Returns:
            Probability of measuring the state
        """
        pass
    
    @abstractmethod
    def is_pure(self) -> bool:
        """Return True if the state is pure, False if mixed."""
        pass
    
    @abstractmethod
    def apply_error_correction(self, correction_type: ErrorCorrectionType) -> 'QuantumState':
        """
        Apply error correction to the quantum state.
        
        Args:
            correction_type: Type of error correction to apply
            
        Returns:
            Error-corrected quantum state
        """
        pass
    
    @abstractmethod
    def to_density_matrix(self) -> 'DensityMatrix':
        """Convert the quantum state to a density matrix representation."""
        pass
    
    @abstractmethod
    def trace_partial(self, qubit_indices: List[int]) -> 'QuantumState':
        """
        Perform a partial trace over the specified qubits.
        
        Args:
            qubit_indices: Indices of qubits to trace out
            
        Returns:
            Reduced quantum state
        """
        pass
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate the fidelity between this state and another quantum state.
        
        Args:
            other: Another quantum state
            
        Returns:
            Fidelity between 0 and 1
        """
        # Default implementation can be overridden by subclasses for efficiency
        dm1 = self.to_density_matrix().get_state_representation()
        dm2 = other.to_density_matrix().get_state_representation()
        
        # Calculate matrix square root
        sqrt_dm1 = np.sqrt(dm1)
        fidelity_matrix = sqrt_dm1 @ dm2 @ sqrt_dm1
        
        # Fidelity is trace of square root
        return float(np.real(np.trace(np.sqrt(fidelity_matrix))))
    
    def entanglement_entropy(self, subsystem_qubits: List[int]) -> float:
        """
        Calculate the entanglement entropy of a subsystem.
        
        Args:
            subsystem_qubits: Qubit indices of the subsystem
            
        Returns:
            von Neumann entropy of the reduced density matrix
        """
        # Get reduced density matrix by tracing out other qubits
        all_qubits = list(range(self.get_num_qubits()))
        traced_qubits = [q for q in all_qubits if q not in subsystem_qubits]
        reduced_state = self.trace_partial(traced_qubits)
        
        # Calculate eigenvalues
        rho = reduced_state.get_state_representation()
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # Remove near-zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy: -sum(λ_i * log2(λ_i))
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


class StateVector(QuantumState):
    """
    Representation of a pure quantum state as a state vector.
    
    This class represents a pure quantum state as a complex vector in the
    computational basis.
    """
    
    def __init__(self, state_vector: np.ndarray = None, num_qubits: int = None):
        """
        Initialize a state vector.
        
        Args:
            state_vector: Complex array of amplitudes. If None, initialize to |0...0⟩
            num_qubits: Number of qubits. Required if state_vector is None.
        """
        if state_vector is not None:
            # Verify valid state vector
            state_vector = np.asarray(state_vector, dtype=complex)
            dim = state_vector.shape[0]
            if not (dim & (dim - 1) == 0):  # Check if dim is a power of 2
                raise ValueError("State vector dimension must be a power of 2")
            
            # Normalize if needed
            norm = np.linalg.norm(state_vector)
            if not np.isclose(norm, 1.0):
                state_vector = state_vector / norm
                
            self._state = state_vector
            self._num_qubits = int(np.log2(dim))
        
        elif num_qubits is not None:
            # Initialize to |0...0⟩
            dim = 2 ** num_qubits
            self._state = np.zeros(dim, dtype=complex)
            self._state[0] = 1.0
            self._num_qubits = num_qubits
        
        else:
            raise ValueError("Either state_vector or num_qubits must be provided")
    
    def get_num_qubits(self) -> int:
        return self._num_qubits
    
    def get_state_representation(self) -> np.ndarray:
        return self._state
    
    def is_pure(self) -> bool:
        return True
    
    def get_probability(self, state_index: int) -> float:
        return np.abs(self._state[state_index]) ** 2
    
    def apply_gate(self, gate: np.ndarray, target_qubits: List[int]) -> None:
        """
        Apply a quantum gate to the specified qubits.
        
        Args:
            gate: Unitary matrix representing the quantum gate
            target_qubits: List of qubit indices the gate acts on
        """
        # Verify target qubits
        if any(q >= self._num_qubits for q in target_qubits):
            raise ValueError("Qubit index out of range")
            
        # Check gate shape matches number of target qubits
        gate_qubits = int(np.log2(gate.shape[0]))
        if gate_qubits != len(target_qubits):
            raise ValueError(f"Gate acts on {gate_qubits} qubits but {len(target_qubits)} provided")
            
        # Identity for other qubits
        if gate_qubits == self._num_qubits:
            # Gate acts on all qubits, simply apply matrix
            self._state = gate @ self._state
            return
            
        # Apply gate using tensor network approach
        # Sort target qubits in ascending order
        sorted_targets = sorted(target_qubits)
        
        # Convert state vector to multi-dimensional array
        state_tensor = self._state.reshape([2] * self._num_qubits)
        
        # Transpose to bring target qubits to the front
        perm = sorted_targets + [q for q in range(self._num_qubits) if q not in sorted_targets]
        inv_perm = np.argsort(perm)
        state_tensor = np.transpose(state_tensor, perm)
        
        # Reshape for gate application
        gate_dim = 2 ** gate_qubits
        other_dim = 2 ** (self._num_qubits - gate_qubits)
        state_mat = state_tensor.reshape((gate_dim, other_dim))
        
        # Apply gate
        state_mat = gate @ state_mat
        
        # Reshape back
        state_tensor = state_mat.reshape([2] * self._num_qubits)
        
        # Transpose back to original order
        state_tensor = np.transpose(state_tensor, inv_perm)
        
        # Reshape to state vector
        self._state = state_tensor.reshape(2 ** self._num_qubits)
    
    def measure(self, qubit_indices: List[int] = None) -> Tuple[int, 'QuantumState']:
        """
        Perform a measurement on the specified qubits.
        
        Args:
            qubit_indices: Indices of qubits to measure. If None, measure all qubits.
            
        Returns:
            Tuple of (measurement outcome, post-measurement state)
        """
        if qubit_indices is None:
            qubit_indices = list(range(self._num_qubits))
        
        if not qubit_indices:
            # No qubits to measure, return copy of current state
            return 0, StateVector(state_vector=self._state.copy())
            
        # Calculate probabilities for each measurement outcome
        probs = {}
        measure_dim = 2 ** len(qubit_indices)
        
        # For each basis state
        for i in range(2 ** self._num_qubits):
            # Extract the measured subsystem bits
            binary = format(i, f"0{self._num_qubits}b")
            measured_bits = ''.join(binary[self._num_qubits - 1 - q] for q in qubit_indices)
            outcome = int(measured_bits, 2)
            
            # Accumulate probability
            prob = self.get_probability(i)
            probs[outcome] = probs.get(outcome, 0) + prob
            
        # Choose outcome based on probabilities
        outcomes = list(probs.keys())
        probabilities = [probs[k] for k in outcomes]
        outcome_idx = np.random.choice(len(outcomes), p=probabilities)
        outcome = outcomes[outcome_idx]
        
        # Prepare post-measurement state
        new_state = np.zeros_like(self._state)
        
        # For each basis state
        for i in range(2 ** self._num_qubits):
            # Extract the measured subsystem bits
            binary = format(i, f"0{self._num_qubits}b")
            measured_bits = ''.join(binary[self._num_qubits - 1 - q] for q in qubit_indices)
            state_outcome = int(measured_bits, 2)
            
            # If this basis state is consistent with the measurement outcome
            if state_outcome == outcome:
                new_state[i] = self._state[i]
                
        # Normalize
        new_state = new_state / np.linalg.norm(new_state)
        
        return outcome, StateVector(state_vector=new_state)
    
    def to_density_matrix(self) -> 'DensityMatrix':
        """Convert the state vector to a density matrix representation."""
        # ρ = |ψ⟩⟨ψ|
        dm = np.outer(self._state, np.conj(self._state))
        return DensityMatrix(density_matrix=dm)
    
    def trace_partial(self, qubit_indices: List[int]) -> 'QuantumState':
        """
        Perform a partial trace over the specified qubits.
        
        Args:
            qubit_indices: Indices of qubits to trace out
            
        Returns:
            Reduced quantum state (always a DensityMatrix)
        """
        # Convert to density matrix first, then trace
        return self.to_density_matrix().trace_partial(qubit_indices)
    
    def apply_error_correction(self, correction_type: ErrorCorrectionType) -> 'QuantumState':
        """
        Apply error correction to the quantum state.
        
        Args:
            correction_type: Type of error correction to apply
            
        Returns:
            Error-corrected quantum state
        """
        if correction_type == ErrorCorrectionType.NONE:
            return self
            
        # For other correction types, we'd implement specific error correction protocols
        # This would typically involve syndrome measurement and correction
        # For simplicity, just return a copy of the current state
        # In a real implementation, this would be expanded significantly
        return StateVector(state_vector=self._state.copy())


class DensityMatrix(QuantumState):
    """
    Representation of a quantum state as a density matrix.
    
    This class can represent both pure and mixed quantum states using the
    density matrix formalism: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
    """
    
    def __init__(self, density_matrix: np.ndarray = None, num_qubits: int = None):
        """
        Initialize a density matrix.
        
        Args:
            density_matrix: Complex square matrix. If None, initialize to |0...0⟩⟨0...0|
            num_qubits: Number of qubits. Required if density_matrix is None.
        """
        if density_matrix is not None:
            # Verify valid density matrix
            density_matrix = np.asarray(density_matrix, dtype=complex)
            if density_matrix.shape[0] != density_matrix.shape[1]:
                raise ValueError("Density matrix must be square")
                
            dim = density_matrix.shape[0]
            if not (dim & (dim - 1) == 0):  # Check if dim is a power of 2
                raise ValueError("Density matrix dimension must be a power of 2")
                
            # Verify Hermitian
            if not np.allclose(density_matrix, density_matrix.conj().T):
                raise ValueError("Density matrix must be Hermitian")
                
            # Verify trace = 1
            if not np.isclose(np.trace(density_matrix), 1.0):
                raise ValueError("Density matrix must have trace 1")
                
            self._state = density_matrix
            self._num_qubits = int(np.log2(dim))
            
        elif num_qubits is not None:
            # Initialize to |0...0⟩⟨0...0|
            dim = 2 ** num_qubits
            self._state = np.zeros((dim, dim), dtype=complex)
            self._state[0,

import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Any
import scipy.linalg as la


class QuantumState:
    """
    Base class for quantum states.
    
    This class serves as the foundation for all quantum state representations,
    providing common functionality and interfaces for state manipulation.
    """
    
    def __init__(self, dims: List[int] = None):
        """
        Initialize a quantum state.
        
        Args:
            dims: List of dimensions for each subsystem
        """
        self.dims = dims or [2]
        self.total_dim = np.prod(self.dims)
        
    def evolve(self, operator: np.ndarray) -> 'QuantumState':
        """
        Evolve the quantum state using the provided operator.
        
        Args:
            operator: Operator to evolve the state with
            
        Returns:
            Evolved quantum state
        """
        raise NotImplementedError("Subclasses must implement evolve")
    
    def measure(self, observable: np.ndarray = None) -> Tuple[Any, 'QuantumState']:
        """
        Measure the quantum state with respect to an observable.
        
        Args:
            observable: Observable to measure (default: measure in computational basis)
            
        Returns:
            Tuple of (measurement result, post-measurement state)
        """
        raise NotImplementedError("Subclasses must implement measure")
    
    def expectation(self, observable: np.ndarray) -> float:
        """
        Calculate the expectation value of an observable.
        
        Args:
            observable: Observable operator
            
        Returns:
            Expectation value
        """
        raise NotImplementedError("Subclasses must implement expectation")
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate the fidelity between this state and another.
        
        Args:
            other: Other quantum state
            
        Returns:
            Fidelity value between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement fidelity")
    
    def partial_trace(self, subsystems: List[int]) -> 'DensityMatrix':
        """
        Perform partial trace over specified subsystems.
        
        Args:
            subsystems: List of subsystem indices to trace out
            
        Returns:
            Reduced density matrix
        """
        raise NotImplementedError("Subclasses must implement partial_trace")
    
    def to_density_matrix(self) -> 'DensityMatrix':
        """
        Convert to density matrix representation.
        
        Returns:
            Density matrix representation of this state
        """
        raise NotImplementedError("Subclasses must implement to_density_matrix")


class StateVector(QuantumState):
    """
    Pure quantum state represented as a state vector.
    
    This class implements a pure quantum state using the state vector formalism,
    providing methods for state manipulation, measurement, and transformation.
    """
    
    def __init__(self, vector: np.ndarray = None, dims: List[int] = None):
        """
        Initialize a state vector.
        
        Args:
            vector: Complex state vector
            dims: List of dimensions for each subsystem
        """
        super().__init__(dims)
        
        if vector is None:
            # Initialize to |0> state by default
            vector = np.zeros(self.total_dim, dtype=complex)
            vector[0] = 1.0
        
        # Ensure vector is properly normalized
        self.vector = vector / np.sqrt(np.sum(np.abs(vector)**2))
        
    def evolve(self, operator: np.ndarray) -> 'StateVector':
        """
        Evolve the state vector using a unitary operator.
        
        Args:
            operator: Unitary operator for evolution
            
        Returns:
            Evolved state vector
        """
        new_vector = operator @ self.vector
        return StateVector(new_vector, self.dims)
    
    def measure(self, observable: np.ndarray = None) -> Tuple[int, 'StateVector']:
        """
        Perform a projective measurement.
        
        Args:
            observable: Observable to measure (default: measure in computational basis)
            
        Returns:
            Tuple of (measurement outcome, post-measurement state)
        """
        if observable is None:
            # Measure in computational basis
            probabilities = np.abs(self.vector)**2
            outcome = np.random.choice(self.total_dim, p=probabilities)
            
            # Create post-measurement state
            new_state = np.zeros_like(self.vector)
            new_state[outcome] = 1.0
            
            return outcome, StateVector(new_state, self.dims)
        else:
            # Diagonalize the observable
            eigenvalues, eigenvectors = np.linalg.eigh(observable)
            
            # Calculate projection probabilities
            projections = eigenvectors.conj().T @ self.vector
            probabilities = np.abs(projections)**2
            
            # Select an outcome based on probabilities
            idx = np.random.choice(len(eigenvalues), p=probabilities)
            outcome = eigenvalues[idx]
            
            # Calculate post-measurement state
            new_state = eigenvectors[:, idx] * (projections[idx] / np.abs(projections[idx]))
            
            return outcome, StateVector(new_state, self.dims)
    
    def expectation(self, observable: np.ndarray) -> float:
        """
        Calculate the expectation value of an observable.
        
        Args:
            observable: Observable operator
            
        Returns:
            Expectation value
        """
        return self.vector.conj() @ observable @ self.vector
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate the fidelity with another quantum state.
        
        Args:
            other: Other quantum state
            
        Returns:
            Fidelity value between 0 and 1
        """
        if isinstance(other, StateVector):
            overlap = np.abs(self.vector.conj() @ other.vector)**2
            return overlap
        else:
            # For mixed states, use the more general definition
            return other.fidelity(self)
    
    def partial_trace(self, subsystems: List[int]) -> 'DensityMatrix':
        """
        Perform partial trace over specified subsystems.
        
        Args:
            subsystems: List of subsystem indices to trace out
            
        Returns:
            Reduced density matrix
        """
        # Convert to density matrix first, then trace
        density_matrix = self.to_density_matrix()
        return density_matrix.partial_trace(subsystems)
    
    def to_density_matrix(self) -> 'DensityMatrix':
        """
        Convert to density matrix representation.
        
        Returns:
            Density matrix representation
        """
        density = np.outer(self.vector, self.vector.conj())
        return DensityMatrix(density, self.dims)
    
    def create_superposition(self, states: List['StateVector'], 
                             amplitudes: List[complex]) -> 'StateVector':
        """
        Create a superposition of quantum states.
        
        Args:
            states: List of state vectors
            amplitudes: Complex amplitudes for each state
            
        Returns:
            Superposition state
        """
        if len(states) != len(amplitudes):
            raise ValueError("Number of states must match number of amplitudes")
        
        # Normalize amplitudes
        norm = np.sqrt(sum(np.abs(a)**2 for a in amplitudes))
        amplitudes = [a/norm for a in amplitudes]
        
        # Create superposition
        superposition = np.zeros(self.total_dim, dtype=complex)
        for state, amplitude in zip(states, amplitudes):
            superposition += amplitude * state.vector
            
        return StateVector(superposition, self.dims)
    
    @staticmethod
    def create_bell_state(bell_type: int = 0) -> 'StateVector':
        """
        Create one of the four Bell states.
        
        Args:
            bell_type: Integer 0-3 selecting which Bell state to create
                0: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                1: |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                2: |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                3: |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
                
        Returns:
            Bell state as a StateVector
        """
        if bell_type == 0:  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            vector = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        elif bell_type == 1:  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            vector = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
        elif bell_type == 2:  # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            vector = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
        elif bell_type == 3:  # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            vector = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
        else:
            raise ValueError("Bell state type must be between 0 and 3")
            
        return StateVector(vector, [2, 2])


class DensityMatrix(QuantumState):
    """
    Mixed quantum state represented as a density matrix.
    
    This class implements a general quantum state using the density matrix formalism,
    handling both pure and mixed states with methods for state manipulation and measurement.
    """
    
    def __init__(self, matrix: np.ndarray = None, dims: List[int] = None):
        """
        Initialize a density matrix.
        
        Args:
            matrix: Complex density matrix
            dims: List of dimensions for each subsystem
        """
        super().__init__(dims)
        
        if matrix is None:
            # Initialize to |0⟩⟨0| state by default
            matrix = np.zeros((self.total_dim, self.total_dim), dtype=complex)
            matrix[0, 0] = 1.0
            
        self.matrix = matrix
        
        # Ensure matrix is properly normalized
        trace = np.trace(matrix)
        if not np.isclose(trace, 1.0):
            self.matrix = matrix / trace
    
    def evolve(self, operator: np.ndarray) -> 'DensityMatrix':
        """
        Evolve the density matrix using a unitary operator.
        
        Args:
            operator: Unitary operator for evolution
            
        Returns:
            Evolved density matrix
        """
        new_matrix = operator @ self.matrix @ operator.conj().T
        return DensityMatrix(new_matrix, self.dims)
    
    def measure(self, observable: np.ndarray = None) -> Tuple[float, 'DensityMatrix']:
        """
        Perform a measurement on the density matrix.
        
        Args:
            observable: Observable to measure (default: measure in computational basis)
            
        Returns:
            Tuple of (measurement outcome, post-measurement state)
        """
        if observable is None:
            # Measure in computational basis
            probabilities = np.diag(self.matrix).real
            outcome = np.random.choice(self.total_dim, p=probabilities)
            
            # Create post-measurement state
            new_matrix = np.zeros_like(self.matrix)
            new_matrix[outcome, outcome] = 1.0
            
            return outcome, DensityMatrix(new_matrix, self.dims)
        else:
            # Diagonalize the observable
            eigenvalues, eigenvectors = np.linalg.eigh(observable)
            
            # Calculate projection probabilities
            projectors = [np.outer(eigenvectors[:, i], eigenvectors[:, i].conj()) 
                         for i in range(len(eigenvalues))]
            
            probabilities = [np.real(np.trace(projector @ self.matrix)) 
                            for projector in projectors]
            
            # Select an outcome based on probabilities
            idx = np.random.choice(len(eigenvalues), p=probabilities)
            outcome = eigenvalues[idx]
            
            # Calculate post-measurement state
            new_matrix = projectors[idx] @ self.matrix @ projectors[idx]
            new_matrix = new_matrix / np.trace(new_matrix)
            
            return outcome, DensityMatrix(new_matrix, self.dims)
    
    def expectation(self, observable: np.ndarray) -> float:
        """
        Calculate the expectation value of an observable.
        
        Args:
            observable: Observable operator
            
        Returns:
            Expectation value
        """
        return np.real(np.trace(observable @ self.matrix))
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate the fidelity with another quantum state.
        
        Args:
            other: Other quantum state
            
        Returns:
            Fidelity value between 0 and 1
        """
        if isinstance(other, StateVector):
            # Convert state vector to density matrix
            other = other.to_density_matrix()
            
        # Calculate fidelity between density matrices
        sqrt_rho = la.sqrtm(self.matrix)
        fid = np.trace(la.sqrtm(sqrt_rho @ other.matrix @ sqrt_rho))**2
        
        # Handle numerical issues (small imaginary parts)
        return np.real(fid)
    
    def partial_trace(self, subsystems: List[int]) -> 'DensityMatrix':
        """
        Perform partial trace over specified subsystems.
        
        Args:
            subsystems: List of subsystem indices to trace out
            
        Returns:
            Reduced density matrix
        """
        # Ensure we're tracing out valid subsystems
        if not all(0 <= s < len(self.dims) for s in subsystems):
            raise ValueError("Invalid subsystem indices")
            
        # Calculate which subsystems to keep
        keep_systems = [i for i in range(len(self.dims)) if i not in subsystems]
        
        if not keep_systems:  # Tracing out everything
            return DensityMatrix(np.array([[1.0]]), [1])
        
        # Calculate dimensions and shapes
        keep_dims = [self.dims[i] for i in keep_systems]
        trace_dims = [self.dims[i] for i in subsystems]
        
        # Reshape density matrix for partial trace
        rho_reshaped = self.matrix.reshape(self.dims + self.dims)
        
        # Initialize result tensor
        result_shape = [self.dims[i] for i in keep_systems] + [self.dims[i] for i in keep_systems]
        result = np.zeros(result_shape, dtype

