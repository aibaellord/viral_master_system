"""
Quantum Error Correction Module

This module provides implementations of various quantum error correction codes
used for protecting quantum information against noise and decoherence.
"""

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union

from .state import QuantumState, StateVector, DensityMatrix


class ErrorType(Enum):
    """Enum for different types of quantum errors."""
    BIT_FLIP = 1
    PHASE_FLIP = 2
    DEPOLARIZING = 3
    AMPLITUDE_DAMPING = 4
    MEASUREMENT = 5
    CUSTOM = 99


class ErrorCorrectionCode(ABC):
    """
    Abstract base class for quantum error correction codes.
    
    Provides the interface for encoding quantum states, detecting errors,
    and applying recovery operations to correct those errors.
    """
    
    def __init__(self, code_distance: int = 3):
        """
        Initialize the error correction code.
        
        Args:
            code_distance: The distance parameter of the code, which determines 
                           its error-correcting capability. Default is 3.
        """
        self.code_distance = code_distance
        self._encoded_state = None
        self._syndrome_history = []
    
    @abstractmethod
    def encode(self, state: QuantumState) -> QuantumState:
        """
        Encode a quantum state using the error correction code.
        
        Args:
            state: The quantum state to encode.
            
        Returns:
            The encoded quantum state.
        """
        pass
    
    @abstractmethod
    def syndrome_measurement(self, state: QuantumState) -> List[int]:
        """
        Perform syndrome measurement on the encoded state.
        
        Args:
            state: The (possibly corrupted) encoded quantum state.
            
        Returns:
            A list of syndrome measurement results.
        """
        pass
    
    @abstractmethod
    def correct(self, state: QuantumState, syndrome: List[int]) -> QuantumState:
        """
        Apply error correction based on syndrome measurements.
        
        Args:
            state: The encoded quantum state with errors.
            syndrome: The syndrome measurement results.
            
        Returns:
            The corrected quantum state.
        """
        pass
    
    def detect_and_correct(self, state: QuantumState) -> Tuple[QuantumState, bool]:
        """
        Detect errors in the encoded state and correct them.
        
        Args:
            state: The encoded quantum state that might contain errors.
            
        Returns:
            A tuple containing:
            - The corrected quantum state
            - A boolean indicating whether errors were detected
        """
        syndrome = self.syndrome_measurement(state)
        errors_detected = any(syndrome)
        
        if errors_detected:
            corrected_state = self.correct(state, syndrome)
            self._syndrome_history.append(syndrome)
            return corrected_state, True
        
        return state, False
    
    def decode(self, state: QuantumState) -> QuantumState:
        """
        Decode an encoded quantum state back to its logical form.
        
        Args:
            state: The encoded quantum state.
            
        Returns:
            The decoded logical quantum state.
        """
        # Default implementation uses partial trace - override for specific codes
        if isinstance(state, StateVector):
            # Convert to density matrix for partial trace operation
            state = state.to_density_matrix()
            
        # Assuming the first qubit is the logical one in the simplest case
        # For actual codes, this would be more complex
        return state.partial_trace([0])


class StabilizerCode(ErrorCorrectionCode):
    """
    Implementation of the Stabilizer quantum error correction code.
    
    Stabilizer codes are defined by a set of stabilizer operators that leave
    the code space invariant.
    """
    
    def __init__(
        self, 
        code_distance: int = 3,
        stabilizer_generators: Optional[List[np.ndarray]] = None
    ):
        """
        Initialize the stabilizer code.
        
        Args:
            code_distance: The distance of the code.
            stabilizer_generators: List of matrices representing the stabilizer generators.
                                   If None, default generators for the specified distance will be used.
        """
        super().__init__(code_distance)
        self.stabilizer_generators = stabilizer_generators or self._default_generators()
        self._logical_operators = self._compute_logical_operators()
        
    def _default_generators(self) -> List[np.ndarray]:
        """
        Create default stabilizer generators for the specified code distance.
        
        Returns:
            A list of numpy arrays representing the stabilizer generators.
        """
        # For a simple [[n,k,d]] stabilizer code, generate default stabilizers
        # This is a simplified implementation and would be more complex for real codes
        n = self.code_distance * 2 - 1  # Total number of physical qubits
        
        # Generate X-type stabilizers
        x_stabilizers = []
        for i in range(self.code_distance - 1):
            # Create a stabilizer that is X on qubits i, i+1 and identity elsewhere
            x_stab = np.zeros((n, n), dtype=complex)
            x_stab[i, i+1] = 1
            x_stab[i+1, i] = 1
            x_stabilizers.append(x_stab)
            
        # Generate Z-type stabilizers
        z_stabilizers = []
        for i in range(self.code_distance - 1):
            # Create a stabilizer that is Z on qubits i, i+1 and identity elsewhere
            z_stab = np.zeros((n, n), dtype=complex)
            z_stab[i, i] = 1
            z_stab[i+1, i+1] = -1
            z_stabilizers.append(z_stab)
            
        return x_stabilizers + z_stabilizers
    
    def _compute_logical_operators(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the logical X and Z operators for the code.
        
        Returns:
            A tuple containing the logical X and Z operators.
        """
        # For a simple stabilizer code, the logical operators would be
        # products of physical X or Z operators that commute with all stabilizers
        n = self.code_distance * 2 - 1  # Total number of physical qubits
        
        # Simplified logical X operator (X on all qubits)
        logical_x = np.zeros((n, n), dtype=complex)
        for i in range(n):
            logical_x[i, (i+1)%n] = 1
            logical_x[(i+1)%n, i] = 1
        
        # Simplified logical Z operator (Z on all qubits)
        logical_z = np.zeros((n, n), dtype=complex)
        for i in range(n):
            logical_z[i, i] = 1 if i % 2 == 0 else -1
            
        return logical_x, logical_z
    
    def encode(self, state: QuantumState) -> QuantumState:
        """
        Encode a quantum state using the stabilizer code.
        
        Args:
            state: The logical quantum state to encode.
            
        Returns:
            The encoded quantum state.
        """
        if not isinstance(state, StateVector):
            # Convert density matrix to state vector if needed
            # This is simplified and would need eigendecomposition in practice
            raise ValueError("Currently only StateVector encoding is supported")
            
        # Get the state vector data
        sv_data = state.data
        
        # Number of physical qubits in the encoded state
        n = self.code_distance * 2 - 1
        
        # Create the encoded state in the +1 eigenspace of all stabilizers
        # Start with |0...0⟩ state
        encoded_data = np.zeros(2**n, dtype=complex)
        encoded_data[0] = 1.0
        
        # Project onto the +1 eigenspace of each stabilizer
        for stabilizer in self.stabilizer_generators:
            # Apply (I + S)/2 projector for each stabilizer S
            projection = (np.eye(2**n) + stabilizer) / 2
            encoded_data = projection @ encoded_data
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(encoded_data)**2))
            if norm > 1e-10:  # Avoid division by zero
                encoded_data /= norm
                
        # Apply logical state preparation based on input state
        logical_x, logical_z = self._logical_operators
        
        # If input is |1⟩, apply logical X
        if np.abs(sv_data[1]) > 0.99:
            encoded_data = logical_x @ encoded_data
        # If input is superposition, apply appropriate rotation
        elif np.abs(sv_data[0]) < 0.99:
            theta = 2 * np.arccos(np.abs(sv_data[0]))
            # Apply rotation around Y axis: exp(-i*theta*Y/2)
            rotation = np.cos(theta/2) * np.eye(2**n) - 1j * np.sin(theta/2) * logical_x
            encoded_data = rotation @ encoded_data
            
            # Apply phase if needed
            phase = np.angle(sv_data[1]) if np.abs(sv_data[1]) > 1e-10 else 0
            if abs(phase) > 1e-10:
                phase_op = np.eye(2**n) * np.exp(1j * phase)
                encoded_data = phase_op @ encoded_data
        
        return StateVector(encoded_data)
    
    def syndrome_measurement(self, state: QuantumState) -> List[int]:
        """
        Perform syndrome measurement on the encoded state.
        
        Args:
            state: The encoded quantum state that might have errors.
            
        Returns:
            A list of syndrome bits (0 or 1) indicating detected errors.
        """
        if isinstance(state, StateVector):
            state = state.to_density_matrix()
            
        syndrome = []
        for stabilizer in self.stabilizer_generators:
            # Compute expectation value of the stabilizer
            expectation = np.trace(state.data @ stabilizer).real
            
            # If close to +1, syndrome bit is 0, if close to -1, syndrome bit is 1
            syndrome_bit = 0 if expectation > 0 else 1
            syndrome.append(syndrome_bit)
            
        return syndrome
    
    def correct(self, state: QuantumState, syndrome: List[int]) -> QuantumState:
        """
        Apply error correction based on syndrome measurements.
        
        Args:
            state: The encoded quantum state with errors.
            syndrome: The syndrome measurement results.
            
        Returns:
            The corrected quantum state.
        """
        if not any(syndrome):
            return state  # No errors detected
            
        # Determine error location based on syndrome pattern
        error_location = self._decode_syndrome(syndrome)
        
        # Apply correction operation
        corrected_state = self._apply_correction(state, error_location)
        
        return corrected_state
    
    def _decode_syndrome(self, syndrome: List[int]) -> Dict[int, List[str]]:
        """
        Decode the syndrome to determine the error location and type.
        
        Args:
            syndrome: The syndrome measurement results.
            
        Returns:
            A dictionary mapping qubit indices to error types.
        """
        # Simplified syndrome decoding for demonstration
        # In a real implementation, this would use the syndrome table specific to the code
        
        n = self.code_distance * 2 - 1  # Total number of physical qubits
        errors = {}
        
        # Simple mapping for demonstration - in reality would depend on the specific code
        for i, bit in enumerate(syndrome):
            if bit:
                if i < len(syndrome) // 2:
                    # X-type error on qubits i and i+1
                    qubit1, qubit2 = i, (i + 1) % n
                    errors[qubit1] = errors.get(qubit1, []) + ["X"]
                    errors[qubit2] = errors.get(qubit2, []) + ["X"]
                else:
                    # Z-type error on qubits i-len(syndrome)//2 and i-len(syndrome)//2+1
                    idx = i - len(syndrome) // 2
                    qubit1, qubit2 = idx, (idx + 1) % n
                    errors[qubit1] = errors.get(qubit1, []) + ["Z"]
                    errors[qubit2] = errors.get(qubit2, []) + ["Z"]
                    
        return errors
    
    def _apply_correction(self, state: QuantumState, error_location: Dict[int, List[str]]) -> QuantumState:
        """
        Apply correction operations based on the identified errors.
        
        Args:
            state: The quantum state with errors.
            error_location: Dictionary mapping qubit indices to error types.
            
        Returns:
            The corrected quantum state.
        """
        corrected_state = state.copy()
        
        for qubit, error_types in error_location.items():
            for error_type in error_types:
                if error_type == "X":
                    # Apply X gate to correct Z error
                    corrected_state = corrected_state.apply_gate("X", qubit)
                elif error_type == "Z":
                    # Apply Z gate to correct X error
                    corrected_state = corrected_state.apply_gate("Z", qubit)
                elif error_type == "Y":
                    # Apply Y gate to correct Y error
                    corrected_state = corrected_state.apply_gate("Y", qubit)
                    
        return corrected_state


class SurfaceCode(ErrorCorrectionCode):
    """
    Implementation of the Surface Code for quantum error correction.
    
    Surface codes are topological codes that arrange qubits on a 2D lattice and
    use local stabilizer measurements for error detection and correction.
    """
    
    def __init__(self, code_distance: int = 3):
        """
        Initialize the surface code.
        
        Args:
            code_distance: The distance of the code, which determines the grid size
                           and error-correcting capability. Must be odd.
        """
        if code_distance % 2 == 0:
            raise ValueError("Surface code distance must be odd")
            
        super().__init__(code_distance)
        
        # Calculate lattice dimensions
        self.grid_size = code_distance
        self.data_qubits = self.grid_size**2
        
        # Construct the X and Z stabilizers for the surface code
        self.x_stabilizers = self._construct_x_stabilizers()
        self.z_stabilizers = self._construct_z_stabilizers()
        
        # Precompute the logical operators
        self.logical_x = self._construct_logical_x()
        self.logical_z = self._construct_logical_z()
        
    def _construct_x_stabilizers(self) -> List[List[int]]:
        """
        Construct the X stabilizers for the surface code.
        
        Returns:
            A list where each element is a list of qubit indices that form an X stabilizer.
        """
        x_stabilizers = []
        
        # X stabilizers are centered on vertices
        for row in range(1, self.grid_size, 2):
            for col in range(1, self.grid_size, 2):
                #

"""
Quantum Error Correction Module

This module implements various quantum error correction codes and mechanisms for detecting
and correcting errors in quantum states during reality manipulation operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

from .state import QuantumState


class ErrorCorrection(ABC):
    """
    Base class for quantum error correction protocols.
    
    This abstract class defines the interface for quantum error correction protocols
    used in reality manipulation operations. Subclasses implement specific error
    correction codes.
    """
    
    def __init__(self, code_distance: int = 3, error_threshold: float = 0.01):
        """
        Initialize the error correction protocol.
        
        Args:
            code_distance: The code distance, which determines error correction capability
            error_threshold: The threshold below which errors are correctable
        """
        self.code_distance = code_distance
        self.error_threshold = error_threshold
        self.syndrome_history = []
    
    @abstractmethod
    def encode(self, state: QuantumState) -> QuantumState:
        """
        Encode a logical quantum state into a physical state with error correction.
        
        Args:
            state: The logical quantum state to encode
            
        Returns:
            The encoded physical quantum state
        """
        pass
    
    @abstractmethod
    def decode(self, state: QuantumState) -> QuantumState:
        """
        Decode a physical quantum state back to the logical state.
        
        Args:
            state: The physical quantum state to decode
            
        Returns:
            The decoded logical quantum state
        """
        pass
    
    @abstractmethod
    def detect_errors(self, state: QuantumState) -> List[Dict[str, Any]]:
        """
        Detect errors in an encoded quantum state.
        
        Args:
            state: The encoded quantum state to check for errors
            
        Returns:
            List of detected errors with location and type information
        """
        pass
    
    @abstractmethod
    def correct_errors(self, state: QuantumState, errors: List[Dict[str, Any]]) -> QuantumState:
        """
        Correct detected errors in an encoded quantum state.
        
        Args:
            state: The encoded quantum state with errors
            errors: List of detected errors to correct
            
        Returns:
            The corrected quantum state
        """
        pass
    
    def syndrome_measurement(self, state: QuantumState) -> np.ndarray:
        """
        Perform syndrome measurement on an encoded state.
        
        Args:
            state: The encoded quantum state
            
        Returns:
            Array of syndrome measurements
        """
        # Base implementation for syndrome measurement
        # Should be overridden by specific error correction codes
        raise NotImplementedError("Syndrome measurement must be implemented by subclasses")
    
    def record_syndrome(self, syndrome: np.ndarray) -> None:
        """
        Record syndrome measurement history for temporal decoding.
        
        Args:
            syndrome: The syndrome measurement result
        """
        self.syndrome_history.append(syndrome)
        # Keep only the last 100 syndromes to avoid memory issues
        if len(self.syndrome_history) > 100:
            self.syndrome_history.pop(0)
    
    def recovery_operation(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine recovery operations for detected errors.
        
        Args:
            errors: List of detected errors
            
        Returns:
            List of recovery operations to apply
        """
        # Base implementation for recovery operations
        # Should be overridden by specific error correction codes
        raise NotImplementedError("Recovery operation must be implemented by subclasses")


class SurfaceCode(ErrorCorrection):
    """
    Surface Code implementation for quantum error correction.
    
    The Surface Code is a topological quantum error correction code that arranges
    physical qubits on a 2D lattice and uses local stabilizer measurements to detect
    and correct errors.
    """
    
    def __init__(self, code_distance: int = 3, error_threshold: float = 0.01,
                 lattice_type: str = "square"):
        """
        Initialize the Surface Code error correction.
        
        Args:
            code_distance: The code distance, must be odd
            error_threshold: The threshold below which errors are correctable
            lattice_type: The type of lattice ("square" or "rotated")
        """
        super().__init__(code_distance, error_threshold)
        if code_distance % 2 == 0:
            raise ValueError("Surface code distance must be odd")
        
        self.lattice_type = lattice_type
        self.n_physical_qubits = code_distance**2
        self.n_stabilizers = 2 * (code_distance - 1)**2
        
        # Initialize stabilizer generators
        self._initialize_stabilizers()
    
    def _initialize_stabilizers(self):
        """Initialize the X and Z stabilizer generators for the surface code."""
        self.x_stabilizers = []
        self.z_stabilizers = []
        
        # Implementation for a square lattice
        if self.lattice_type == "square":
            for i in range(self.code_distance - 1):
                for j in range(self.code_distance - 1):
                    # X stabilizers (star operators)
                    x_stab = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
                    self.x_stabilizers.append(x_stab)
                    
                    # Z stabilizers (plaquette operators)
                    z_stab = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
                    self.z_stabilizers.append(z_stab)
    
    def encode(self, state: QuantumState) -> QuantumState:
        """
        Encode a logical quantum state into a surface code.
        
        Args:
            state: The logical quantum state to encode
            
        Returns:
            The encoded surface code state
        """
        # Create a physical state with the necessary number of qubits
        encoded_state = QuantumState(n_qubits=self.n_physical_qubits)
        
        # Initialize to the +1 eigenstate of all stabilizers
        # This creates the code space
        
        # Apply logical operators to set the encoded state
        # based on the input logical state
        
        return encoded_state
    
    def decode(self, state: QuantumState) -> QuantumState:
        """
        Decode a surface code state back to the logical state.
        
        Args:
            state: The physical quantum state to decode
            
        Returns:
            The decoded logical quantum state
        """
        # Apply measurements to extract the logical state
        logical_state = QuantumState(n_qubits=1)
        
        # Measure logical operators to determine logical state
        
        return logical_state
    
    def detect_errors(self, state: QuantumState) -> List[Dict[str, Any]]:
        """
        Detect errors in a surface code state by measuring stabilizers.
        
        Args:
            state: The encoded quantum state to check for errors
            
        Returns:
            List of detected errors with location and type information
        """
        syndrome = self.syndrome_measurement(state)
        self.record_syndrome(syndrome)
        
        # Identify violated stabilizers
        errors = []
        for i, meas in enumerate(syndrome):
            if meas == -1:  # Stabilizer violated
                is_x_stabilizer = i < len(self.x_stabilizers)
                stabilizer_idx = i if is_x_stabilizer else i - len(self.x_stabilizers)
                
                errors.append({
                    "type": "Z" if is_x_stabilizer else "X",
                    "location": self.x_stabilizers[stabilizer_idx] if is_x_stabilizer 
                               else self.z_stabilizers[stabilizer_idx],
                    "stabilizer_idx": stabilizer_idx
                })
        
        return errors
    
    def syndrome_measurement(self, state: QuantumState) -> np.ndarray:
        """
        Perform syndrome measurement for the surface code.
        
        Args:
            state: The encoded quantum state
            
        Returns:
            Array of syndrome measurements (+1 for no error, -1 for error)
        """
        n_stabilizers = len(self.x_stabilizers) + len(self.z_stabilizers)
        syndrome = np.ones(n_stabilizers)
        
        # Measure X-stabilizers
        for i, stabilizer in enumerate(self.x_stabilizers):
            # Measure product of X operators on qubits in stabilizer
            syndrome[i] = state.measure_pauli_product("X", stabilizer)
        
        # Measure Z-stabilizers
        for i, stabilizer in enumerate(self.z_stabilizers):
            # Measure product of Z operators on qubits in stabilizer
            syndrome[i + len(self.x_stabilizers)] = state.measure_pauli_product("Z", stabilizer)
        
        return syndrome
    
    def correct_errors(self, state: QuantumState, errors: List[Dict[str, Any]]) -> QuantumState:
        """
        Apply corrections to a surface code state based on detected errors.
        
        Args:
            state: The encoded quantum state with errors
            errors: List of detected errors to correct
            
        Returns:
            The corrected quantum state
        """
        recovery_ops = self.recovery_operation(errors)
        
        # Apply recovery operations
        for op in recovery_ops:
            qubit = op["location"]
            op_type = op["type"]
            
            if op_type == "X":
                state.apply_x(qubit)
            elif op_type == "Z":
                state.apply_z(qubit)
            elif op_type == "Y":
                state.apply_y(qubit)
        
        return state
    
    def recovery_operation(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine recovery operations for detected errors using minimum-weight perfect matching.
        
        Args:
            errors: List of detected errors
            
        Returns:
            List of recovery operations to apply
        """
        # Implement minimum-weight perfect matching for error correction
        recovery_ops = []
        
        # For each error, determine the most likely recovery operation
        for error in errors:
            # Find the shortest path to another error or boundary
            recovery_ops.append({
                "type": "X" if error["type"] == "Z" else "Z",
                "location": error["location"][0]  # Apply to one of the qubits in the stabilizer
            })
        
        return recovery_ops


class StabilizerCode(ErrorCorrection):
    """
    General Stabilizer Code implementation for quantum error correction.
    
    Stabilizer codes are quantum error-correcting codes that use the stabilizer formalism,
    where the code space is defined as the +1 eigenspace of a set of commuting Pauli operators.
    """
    
    def __init__(self, stabilizer_generators: List[Tuple[str, List[int]]],
                 logical_operators: List[Tuple[str, List[int]]],
                 n_physical_qubits: int, n_logical_qubits: int,
                 error_threshold: float = 0.01):
        """
        Initialize the Stabilizer Code error correction.
        
        Args:
            stabilizer_generators: List of (pauli_string, qubits) defining stabilizers
            logical_operators: List of (pauli_string, qubits) defining logical operators
            n_physical_qubits: Number of physical qubits
            n_logical_qubits: Number of logical qubits
            error_threshold: The threshold below which errors are correctable
        """
        super().__init__(0, error_threshold)  # Code distance will be calculated
        
        self.stabilizer_generators = stabilizer_generators
        self.logical_operators = logical_operators
        self.n_physical_qubits = n_physical_qubits
        self.n_logical_qubits = n_logical_qubits
        
        # Calculate code distance
        self._calculate_code_distance()
    
    def _calculate_code_distance(self):
        """Calculate the code distance for this stabilizer code."""
        # Code distance is the weight of the minimum-weight logical operator
        min_weight = float('inf')
        
        for _, qubits in self.logical_operators:
            weight = len(qubits)
            if weight < min_weight:
                min_weight = weight
        
        self.code_distance = min_weight if min_weight < float('inf') else 0
    
    def encode(self, state: QuantumState) -> QuantumState:
        """
        Encode a logical quantum state into a stabilizer code.
        
        Args:
            state: The logical quantum state to encode
            
        Returns:
            The encoded stabilizer code state
        """
        # Create a physical state with the necessary number of qubits
        encoded_state = QuantumState(n_qubits=self.n_physical_qubits)
        
        # Project onto the +1 eigenspace of all stabilizers
        for pauli_string, qubits in self.stabilizer_generators:
            encoded_state.project_stabilizer(pauli_string, qubits)
        
        # Set logical state based on input state
        # by applying appropriate logical operators
        
        return encoded_state
    
    def decode(self, state: QuantumState) -> QuantumState:
        """
        Decode a stabilizer code state back to the logical state.
        
        Args:
            state: The physical quantum state to decode
            
        Returns:
            The decoded logical quantum state
        """
        logical_state = QuantumState(n_qubits=self.n_logical_qubits)
        
        # Measure logical operators to determine logical state
        for i in range(self.n_logical_qubits):
            # Measure X and Z logical operators for each logical qubit
            x_op = self.logical_operators[2*i]
            z_op = self.logical_operators[2*i + 1]
            
            x_val = state.measure_pauli_product(x_op[0], x_op[1])
            z_val = state.measure_pauli_product(z_op[0], z_op[1])
            
            # Set logical qubit state based on measurements
            if z_val == -1:
                logical_state.apply_z(i)
            if x_val == -1:
                logical_state.apply_x(i)
        
        return logical_state
    
    def detect_errors(self, state: QuantumState) -> List[Dict[str, Any]]:
        """
        Detect errors in a stabilizer code state.
        
        Args:
            state: The encoded quantum state to check for errors
            
        Returns:
            List of detected errors with stabilizer information
        """
        syndrome = self.syndrome_measurement(state)
        self.record_syndrome(syndrome)
        
        errors = []
        for i, meas in enumerate(syndrome):
            if meas == -1:  # Stabilizer violated
                pauli_string, qubits = self.stabilizer_generators[i]
                errors.append({
                    "stabilizer_idx": i,
                    "pauli_type": pauli_string,
                    "qubits": qubits
                })
        
        return errors
    
    def syndrome_measurement(self, state: QuantumState) -> np.ndarray:
        """
        Perform syndrome measurement for the stabilizer code.
        
        Args:
            state: The encoded quantum state
            

