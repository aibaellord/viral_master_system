"""
Quantum Computing Module

This module implements quantum computing primitives and operations for quantum information processing.
It provides classes for quantum state manipulation, quantum gates, error correction, and quantum circuits.

The module is designed to integrate with both simulation environments and real quantum hardware.
"""

from enum import Enum, auto
import numpy as np

# Import core quantum classes
from .state import QuantumState, StateVector, DensityMatrix
from .engine import QuantumEngine
from .error_correction import ErrorCorrection, SurfaceCode, StabilizerCode

# Error correction types enum
class ErrorCorrectionType(Enum):
    """Enumeration of supported error correction types."""
    NONE = auto()
    REPETITION = auto()
    STEANE = auto()
    SHOR = auto()
    SURFACE = auto()
    STABILIZER = auto()
    QUANTUM_LDPC = auto()

# Common quantum gates as numpy arrays
class Gates:
    """Common quantum gates implemented as numpy arrays."""
    # Pauli gates
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)  # NOT gate
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Hadamard gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    # Phase gates
    S = np.array([[1, 0], [0, 1j]], dtype=complex)  # π/2 phase (S gate)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # π/4 phase (T gate)
    
    # Controlled-NOT gate (2-qubit)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # SWAP gate (2-qubit)
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    
    # Toffoli gate (3-qubit controlled-controlled-NOT)
    TOFFOLI = np.zeros((8, 8), dtype=complex)
    TOFFOLI[0:6, 0:6] = np.eye(6)
    TOFFOLI[6:8, 6:8] = np.array([[0, 1], [1, 0]])

# Initialize quantum environment
def initialize_quantum_environment(use_gpu=False, error_correction=ErrorCorrectionType.SURFACE):
    """
    Initialize the quantum computing environment with specified settings.
    
    Args:
        use_gpu (bool): Whether to use GPU acceleration if available
        error_correction (ErrorCorrectionType): Type of error correction to use
        
    Returns:
        QuantumEngine: Initialized quantum engine
    """
    return QuantumEngine(use_gpu=use_gpu, error_correction=error_correction)

# Export public interface
__all__ = [
    'QuantumState', 'StateVector', 'DensityMatrix',
    'QuantumEngine', 'ErrorCorrection', 'SurfaceCode', 'StabilizerCode',
    'ErrorCorrectionType', 'Gates', 'initialize_quantum_environment'
]

"""
Quantum Computing Engine Module

This module provides quantum computing functionalities for reality manipulation,
quantum state processing, and quantum-classical hybrid computing operations.

The quantum engine enables:
- Quantum state preparation and manipulation
- Quantum circuit operations
- Reality fabric interactions
- Quantum-classical hybrid processing
- State optimization and coherence management
- Error correction mechanisms

Usage:
    from core.quantum import QuantumEngine, QuantumState
    
    # Initialize quantum engine
    engine = QuantumEngine(dimensions=8, precision=0.001)
    
    # Create quantum state
    state = QuantumState(dimensions=8)
    
    # Apply quantum operations
    engine.apply_transformation(state, transformation_matrix)
    
    # Measure quantum state
    result = engine.measure(state)
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# Export main classes
from .engine import (
    QuantumEngine,
    QuantumCircuit,
    QuantumGate,
    QuantumMeasurement,
    QuantumRegister,
)

# Export state management
from .state import (
    QuantumState,
    StateVector,
    DensityMatrix,
    Superposition,
    Entanglement,
)

# Export error correction mechanisms
from .error_correction import (
    ErrorCorrection,
    SurfaceCode,
    StabilizerCode,
)

# Quantum state types enum
class QuantumStateType(Enum):
    """Enumeration of quantum state types supported by the engine."""
    PURE = auto()
    MIXED = auto()
    ENTANGLED = auto()
    SUPERPOSITION = auto()
    COHERENT = auto()
    SQUEEZED = auto()

# Quantum operation types enum
class QuantumOperationType(Enum):
    """Enumeration of quantum operations supported by the engine."""
    UNITARY = auto()
    MEASUREMENT = auto()
    RESET = auto()
    NOISE = auto()
    CUSTOM = auto()

# Reality fabric interaction types
class RealityFabricInteraction(Enum):
    """Enumeration of reality fabric interaction types."""
    OBSERVATION = auto()
    MANIPULATION = auto()
    SYNCHRONIZATION = auto()
    ENHANCEMENT = auto()
    DISRUPTION = auto()

# Default engine configuration
DEFAULT_CONFIG = {
    "dimensions": 8,
    "precision": 0.001,
    "max_qubits": 32,
    "error_correction": True,
    "reality_manipulation_strength": 0.85,
    "coherence_threshold": 0.72,
    "gpu_acceleration": True,
}

# Version information
__version__ = "1.0.0"

# Initialize module-level variables
_engine_instance = None

def get_engine(config: Optional[Dict] = None) -> 'QuantumEngine':
    """
    Get or create the global quantum engine instance.
    
    Args:
        config: Optional configuration dictionary to override defaults
        
    Returns:
        QuantumEngine: The global quantum engine instance
    """
    global _engine_instance
    
    if _engine_instance is None:
        # Create new engine with provided or default config
        settings = DEFAULT_CONFIG.copy()
        if config:
            settings.update(config)
        
        from .engine import QuantumEngine
        _engine_instance = QuantumEngine(**settings)
    
    return _engine_instance

# Clean up namespace
del auto, Dict, List, Optional, Tuple, Union, Enum

