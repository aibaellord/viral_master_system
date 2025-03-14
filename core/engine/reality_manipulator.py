import os
import time
import logging
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

# Try to import CUDA-related libraries with proper error handling
try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    logger.warning("PyTorch or CUDA libraries not available. Falling back to CPU processing.")
    HAS_CUDA = False

# Try to import quantum computing libraries with proper error handling
try:
    from qiskit import QuantumCircuit, Aer, execute
    HAS_QUANTUM = True
except ImportError:
    logger.warning("Quantum computing libraries not available. Using classical simulation.")
    HAS_QUANTUM = False


class RealityManipulator:
    """
    Provides reality fabric manipulation capabilities with:
    - CUDA-optimized tensor operations
    - Quantum state integration
    - Performance monitoring
    - Comprehensive error handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Reality Manipulator with optional configuration.
        
        Args:
            config: Configuration dictionary containing parameters for reality manipulation
        """
        self.config = config or {}
        
        # Initialize performance metrics
        self.metrics = {
            'operations_count': 0,
            'total_processing_time': 0,
            'quantum_operations': 0,
            'cuda_operations': 0,
            'errors': 0
        }
        
        # Initialize CUDA device
        self.device = self._setup_cuda_device()
        
        # Initialize quantum backend
        self.quantum_backend = self._initialize_quantum_backend()
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Reality Manipulator initialized. CUDA: {HAS_CUDA}, Quantum: {HAS_QUANTUM}")
    
    def _setup_cuda_device(self) -> Union[torch.device, str]:
        """Set up CUDA device with error handling."""
        try:
            if not HAS_CUDA:
                return "cpu"
            
            # Get device ID from config or default to 0
            device_id = self.config.get('cuda_device_id', 0)
            
            # Check if specified device is available
            if device_id >= torch.cuda.device_count():
                logger.warning(f"Requested CUDA device {device_id} not available. "
                              f"Found {torch.cuda.device_count()} devices. Using device 0.")
                device_id = 0
                
            # Set up device
            device = torch.device(f"cuda:{device_id}")
            
            # Log device info
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
            
            # Set memory management options
            torch.cuda.set_per_process_memory_fraction(
                self.config.get('memory_fraction', 0.8), device_id
            )
            
            return device
        except Exception as e:
            logger.error(f"CUDA setup error: {str(e)}")
            return "cpu"
    
    def _initialize_quantum_backend(self):
        """Initialize quantum backend with error handling."""
        if not HAS_QUANTUM:
            return None
            
        try:
            # Get backend type from config or default to qasm_simulator
            backend_name = self.config.get('quantum_backend', 'qasm_simulator')
            return Aer.get_backend(backend_name)
        except Exception as e:
            logger.error(f"Quantum backend initialization error: {str(e)}")
            return None
    
    def manipulate_reality_fabric(self, 
                                 tensor_data: np.ndarray, 
                                 manipulation_strength: float = 0.5) -> np.ndarray:
        """
        Manipulate reality fabric using tensor operations.
        
        Args:
            tensor_data: Input data representing reality fabric state
            manipulation_strength: Strength of manipulation (0.0 to 1.0)
            
        Returns:
            Manipulated reality fabric state
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.metrics['operations_count'] += 1
                
                # Normalize manipulation strength
                manipulation_strength = max(0.0, min(1.0, manipulation_strength))
                
                # Convert to tensor and move to appropriate device
                if HAS_CUDA:
                    try:
                        tensor = torch.tensor(tensor_data, device=self.device)
                        self.metrics['cuda_operations'] += 1
                    except RuntimeError as e:
                        logger.warning(f"CUDA error: {str(e)}. Falling back to CPU.")
                        tensor = torch.tensor(tensor_data)
                else:
                    tensor = torch.tensor(tensor_data)
                
                # Apply reality manipulation operations
                manipulated_tensor = self._apply_manipulation_operations(tensor, manipulation_strength)
                
                # Convert back to numpy array
                result = manipulated_tensor.cpu().numpy()
                
                # Update metrics
                self.metrics['total_processing_time'] += time.time() - start_time
                
                return result
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Reality manipulation error: {str(e)}")
            # Return original data in case of error
            return tensor_data
    
    def _apply_manipulation_operations(self, 
                                      tensor: torch.Tensor, 
                                      strength: float) -> torch.Tensor:
        """
        Apply core reality manipulation operations to the tensor.
        
        Args:
            tensor: Input tensor
            strength: Manipulation strength
            
        Returns:
            Manipulated tensor
        """
        # Apply non-linear transformation
        transformed = torch.tanh(tensor * strength)
        
        # Apply reality fabric tension
        tension = self.config.get('fabric_tension', 0.7)
        tension_applied = transformed * tension + tensor * (1 - tension)
        
        # Add quantum fluctuations if quantum computing is available
        if HAS_QUANTUM and self.quantum_backend:
            qfluctuations = self._generate_quantum_fluctuations(tensor.shape)
            qfluctuations_tensor = torch.tensor(qfluctuations, device=tensor.device)
            quantum_influence = self.config.get('quantum_influence', 0.3)
            result = tension_applied * (1 - quantum_influence) + qfluctuations_tensor * quantum_influence
            self.metrics['quantum_operations'] += 1
        else:
            # Add classical pseudo-random fluctuations as fallback
            random_fluctuations = torch.rand_like(tensor) * 0.1
            result = tension_applied + (random_fluctuations - 0.05) * strength
        
        return result
    
    def _generate_quantum_fluctuations(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate quantum fluctuations using quantum circuits.
        
        Args:
            shape: Shape of the tensor to generate fluctuations for
            
        Returns:
            Array of quantum fluctuations
        """
        try:
            # Calculate total number of elements
            total_elements = np.prod(shape)
            
            # Create quantum circuit with appropriate number of qubits
            num_qubits = min(10, max(4, int(np.log2(total_elements))))
            circuit = QuantumCircuit(num_qubits)
            
            # Apply quantum operations
            for i in range(num_qubits):
                circuit.h(i)  # Hadamard gate for superposition
                
            # Add entanglement
            for i in range(num_qubits-1):
                circuit.cx(i, i+1)  # CNOT gates for entanglement
            
            # Add measurement
            circuit.measure_all()
            
            # Execute circuit
            job = execute(circuit, self.quantum_backend, shots=total_elements)
            result = job.result()
            
            # Process results into fluctuations
            counts = result.get_counts()
            fluctuations = np.zeros(total_elements)
            
            # Convert bit strings to values between -0.5 and 0.5
            idx = 0
            for bitstring, count in counts.items():
                value = int(bitstring, 2) / (2**num_qubits) - 0.5
                for _ in range(count):
                    if idx < total_elements:
                        fluctuations[idx] = value
                        idx += 1
            
            # Reshape to match the input tensor shape
            return fluctuations.reshape(shape)
        except Exception as e:
            logger.error(f"Quantum fluctuation generation error: {str(e)}")
            # Return small random fluctuations as fallback
            return np.random.uniform(-0.1, 0.1, size=shape)
    
    def integrate_quantum_state(self, quantum_state: np.ndarray, reality_state: np.ndarray) -> np.ndarray:
        """
        Integrate quantum state with reality fabric state.
        
        Args:
            quantum_state: Quantum state array
            reality_state: Reality fabric state array
            
        Returns:
            Integrated state
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.metrics['operations_count'] += 1
                
                # Convert to tensors
                if HAS_CUDA:
                    try:
                        q_tensor = torch.tensor(quantum_state, device=self.device)
                        r_tensor = torch.tensor(reality_state, device=self.device)
                        self.metrics['cuda_operations'] += 1
                    except RuntimeError as e:
                        logger.warning(f"CUDA error in quantum integration: {str(e)}. Falling back to CPU.")
                        q_tensor = torch.tensor(quantum_state)
                        r_tensor = torch.tensor(reality_state)
                else:
                    q_tensor = torch.tensor(quantum_state)
                    r_tensor = torch.tensor(reality_state)
                
                # Calculate coherence factor based on similarity
                coherence = torch.cosine_similarity(q_tensor.flatten(), r_tensor.flatten(), dim=0)
                coherence = (coherence + 1) / 2  # Normalize to [0, 1]
                
                # Apply integration based on coherence
                integration_strength = self.config.get('integration_strength', 0.6)
                integrated = r_tensor * (1 - integration_strength * coherence) + q_tensor * (integration_strength * coherence)
                
                # Normalize result
                if self.config.get('normalize_output', True):
                    integrated = (integrated - integrated.min()) / (integrated.max() - integrated.min() + 1e-8)
                
                # Update metrics
                self.metrics['total_processing_time'] += time.time() - start_time
                
                return integrated.cpu().numpy()
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Quantum state integration error: {str(e)}")
            # Return reality state in case of error
            return reality_state
    
    def optimize_reality_fabric(self, 
                              fabric_state: np.ndarray, 
                              target_state: np.ndarray,
                              iterations: int = 100) -> np.ndarray:
        """
        Optimize reality fabric to approach a target state.
        
        Args:
            fabric_state: Current reality fabric state
            target_state: Target reality state
            iterations: Number of optimization iterations
            
        Returns:
            Optimized reality fabric state
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.metrics['operations_count'] += 1
                
                # Convert to tensors
                if HAS_CUDA:
                    try:
                        fabric_tensor = torch.tensor(fabric_state, device=self.device, requires_grad=True)
                        target_tensor = torch.tensor(target_state, device=self.device)
                        self.metrics['cuda_operations'] += 1
                    except RuntimeError as e:
                        logger.warning(f"CUDA error in optimization: {str(e)}. Falling back to CPU.")
                        fabric_tensor = torch.tensor(fabric_state, requires_grad=True)
                        target_tensor = torch.tensor(target_state)
                else:
                    fabric_tensor = torch.tensor(fabric_state, requires_grad=True)
                    target_tensor = torch.tensor(target_state)
                
                # Create optimizer
                learning_rate = self.config.get('optimization_learning_rate', 0.01)
                optimizer = torch.optim.Adam([fabric_tensor], lr=learning_rate)
                
                # Optimization loop
                for i in range(iterations):
                    # Calculate loss
                    loss = nn.functional.mse_loss(fabric_tensor, target_tensor)
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Early stopping if loss is small enough
                    if loss.item() < self.config.get('early_stopping_threshold', 1e-5):
                        logger.info(f"Optimization converged after {i+1} iterations")
                        break
                
                # Get optimized state
                optimized_state = fabric_tensor.detach().cpu().numpy()
                
                # Update metrics
                self.metrics['total_processing_time'] += time.time() - start_time
                
                return optimized_state
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Reality optimization error: {str(e)}")
            # Return original fabric state in case of error
            return fabric_state
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for reality manipulation operations.
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            metrics_copy = self.metrics.copy()
            
            # Add derived metrics
            if metrics_copy['operations_count'] > 0:
                metrics_copy['avg_processing_time'] = (
                    metrics_copy['total_processing_time'] / 
                    metrics_copy['operations_count']
                )
            
            # Add device information
            if HAS_CUDA:
                metrics_copy['cuda_device_name'] = torch.cuda.get_device_name(
                    self.device.index if self.device != "cpu" else 0
                )
                metrics_copy['cuda_memory_allocated'] = torch.cuda.memory_allocated(
                    self.device.index if self.device != "cpu" else 0
                )
                metrics_copy['cuda_memory_reserved'] = torch.cuda.memory_reserved(
                    self.device.index if self.device != "cpu" else 0
                )
            
            return metrics_copy
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self.lock:
            for key in self.metrics:
                self.metrics[key] = 0
            logger.info("Reality Manipulator metrics reset")

import logging
import time
import threading
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import os
from functools import lru_cache
from collections import OrderedDict

from core.base_component import BaseComponent
try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    from torch.multiprocessing import Pool, Process, set_start_method
    from transformers import pipeline
    HAS_TORCH = True
    
    # Check CUDA availability
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
    
    # Try to initialize CUDA for multi-processing
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        # Already initialized, ignore
        pass
        
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0
    
try:
    # Import for quantum processing if available
    import qiskit
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.visualization import plot_state_city
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False
@dataclass
class QuantumState:
    """Representation of a quantum state for reality manipulation."""
    state_vector: np.ndarray
    entanglement_degree: float
    coherence_factor: float
    collapse_probability: float
    dimensions: int = 1
    
    def is_stable(self) -> bool:
        """Check if quantum state is stable for manipulation."""
        return self.coherence_factor > 0.7 and self.collapse_probability < 0.3

@dataclass
class PerceptionMetrics:
    """Metrics for measuring perception manipulation effectiveness."""
    reality_distortion_factor: float
    psychological_impact: float
    emotional_resonance: float
    attention_capture: float
    believability_score: float
    conviction_strength: float
    narrative_coherence: float
    perception_shift: float
    quantum_alignment: float = 0.0
    dimensional_resonance: float = 0.0
class QuantumNeuralBridge:
    """Bridge between quantum states and neural processing."""
    
    def __init__(self, dimensions: int = 3, device: str = "cpu"):
        self.dimensions = max(1, dimensions)  # At least 1-dimensional
        self.device = device
        self.logger = logging.getLogger("ViralSystem.QuantumNeuralBridge")
        self.quantum_enabled = HAS_QUANTUM
        self.neural_enabled = HAS_TORCH
        self.state_cache = OrderedDict()  # LRU cache for quantum states
        self.cache_size = 100  # Maximum number of cached states
        
        # Initialize quantum simulator if available
        if self.quantum_enabled:
            self.quantum_simulator = Aer.get_backend('statevector_simulator')
        
        self.logger.info(f"QuantumNeuralBridge initialized with {dimensions} dimensions")
        
    def create_quantum_circuit(self, qubits: int = 3) -> Any:
        """Create a quantum circuit for reality manipulation."""
        if not self.quantum_enabled:
            return None
            
        try:
            circuit = QuantumCircuit(qubits)
            
            # Apply basic quantum operations to create superposition
            for i in range(qubits):
                circuit.h(i)  # Hadamard gate for superposition
                
            # Create entanglement
            for i in range(qubits-1):
                circuit.cx(i, i+1)  # CNOT gate for entanglement
                
            return circuit
        except Exception as e:
            self.logger.error(f"Error creating quantum circuit: {str(e)}")
            return None
    
    @lru_cache(maxsize=32)
    def compute_quantum_state(self, content_hash: str, dimensions: int = None) -> QuantumState:
        """Compute quantum state for content, with caching."""
        if dimensions is None:
            dimensions = self.dimensions
            
        # Check cache first
        if content_hash in self.state_cache:
            self.logger.debug(f"Quantum state cache hit for {content_hash[:8]}")
            return self.state_cache[content_hash]
            
        # Generate new state
        if not self.quantum_enabled:
            # Fallback to classical simulation if quantum libraries not available
            state_vector = np.random.rand(2**3)
            state_vector = state_vector / np.linalg.norm(state_vector)
            
            state = QuantumState(
                state_vector=state_vector,
                entanglement_degree=np.random.random(),
                coherence_factor=np.random.random(),
                collapse_probability=np.random.random() * 0.5,
                dimensions=dimensions
            )
        else:
            # Use actual quantum simulation
            try:
                circuit = self.create_quantum_circuit(qubits=dimensions+2)
                if circuit:
                    result = execute(circuit, self.quantum_simulator).result()
                    state_vector = result.get_statevector()
                    
                    state = QuantumState(
                        state_vector=np.array(state_vector),
                        entanglement_degree=np.random.random() * 0.8 + 0.2,  # Higher entanglement
                        coherence_factor=np.random.random() * 0.6 + 0.4,    # Good coherence
                        collapse_probability=np.random.random() * 0.3,      # Low collapse probability
                        dimensions=dimensions
                    )
                else:
                    raise Exception("Failed to create quantum circuit")
            except Exception as e:
                self.logger.error(f"Quantum state computation error: {str(e)}")
                # Fallback to classical simulation
                state_vector = np.random.rand(2**3)
                state_vector = state_vector / np.linalg.norm(state_vector)
                
                state = QuantumState(
                    state_vector=state_vector,
                    entanglement_degree=np.random.random(),
                    coherence_factor=np.random.random(),
                    collapse_probability=np.random.random() * 0.5,
                    dimensions=dimensions
                )
        
        # Update cache, removing oldest item if cache is full
        if len(self.state_cache) >= self.cache_size:
            self.state_cache.popitem(last=False)
        self.state_cache[content_hash] = state
        
        return state
        
    def apply_quantum_transformation(self, content: Dict[str, Any], 
                                    quantum_state: QuantumState) -> Dict[str, Any]:
        """Apply quantum-based transformation to content."""
        if not quantum_state.is_stable():
            self.logger.warning("Unstable quantum state, limiting transformation")
            return content
            
        try:
            if "text" in content:
                # Apply quantum-influenced transformation
                text = content["text"]
                
                # Compute transformation factor based on quantum state
                entanglement_factor = quantum_state.entanglement_degree
                coherence_factor = quantum_state.coherence_factor
                transformation_power = entanglement_factor * coherence_factor
                
                # Apply transformation based on the quantum state properties
                if transformation_power > 0.7:
                    # Enhanced transformation for high-power quantum states
                    content["quantum_enhanced"] = True
                    content["quantum_power"] = transformation_power
                    content["dimensional_depth"] = quantum_state.dimensions
                
            return content
        except Exception as e:
            self.logger.error(f"Quantum transformation error: {str(e)}")
            return content
            
    def neural_quantum_process(self, neural_output: Any, quantum_state: QuantumState) -> Any:
        """Process neural output through quantum lens."""
        if not self.neural_enabled or neural_output is None:
            return neural_output
            
        try:
            # Apply quantum state to influence neural processing
            # This is a simplified implementation
            if isinstance(neural_output, dict) and quantum_state.is_stable():
                neural_output["quantum_influenced"] = True
                neural_output["quantum_dimensions"] = quantum_state.dimensions
            return neural_output
        except Exception as e:
            self.logger.error(f"Neural-quantum processing error: {str(e)}")
            return neural_output

class PerceptionModel:
    """Model for content perception manipulation."""
    
    def __init__(self, model_type: str, config: Dict[str, Any], device: str = "cpu"):
        self.model_type = model_type
        self.config = config
        self.device = device
        self.model = None
        self.logger = logging.getLogger(f"ViralSystem.PerceptionModel.{model_type}")
        
        # CUDA optimization settings
        self.optimize_for_cuda = config.get("optimize_for_cuda", CUDA_AVAILABLE)
        self.precision = config.get("precision", "float32")
        self.batch_size = config.get("batch_size", 1)
        
        # Quantum bridge settings
        self.use_quantum = config.get("use_quantum", HAS_QUANTUM)
        self.quantum_dimensions = config.get("quantum_dimensions", 3)
        
        # Initialize quantum bridge if needed
        if self.use_quantum:
            self.quantum_bridge = QuantumNeuralBridge(
                dimensions=self.quantum_dimensions,
                device=self.device
            )
        
    def load(self) -> bool:
        """Load the perception model."""
        if not HAS_TORCH:
            self.logger.error("PyTorch and transformers are required for perception models")
            return False
            
        try:
            # Configure CUDA optimization if available
            if self.optimize_for_cuda and CUDA_AVAILABLE:
                # Use hardware-optimized settings when available
                device_id = self.config.get("cuda_device_id", 0) % max(1, CUDA_DEVICE_COUNT)
                self.device = f"cuda:{device_id}"
                
                # Set precision based on configuration
                if self.precision == "float16" and torch.cuda.is_available():
                    # Enable automatic mixed precision for better performance
                    self.logger.info(f"Using mixed precision for {self.model_type}")
                    torch.cuda.amp.autocast(enabled=True)
                
                # Configure GPU memory usage
                if self.config.get("optimize_memory", True):
                    torch.cuda.set_per_process_memory_fraction(
                        self.config.get("memory_fraction", 0.8), device_id
                    )
                    
                # Enable optimizations
                torch.backends.cudnn.benchmark = self.config.get("cudnn_benchmark", True)
                
                self.logger.info(f"CUDA optimization enabled for {self.model_type} on {self.device}")
            else:
                self.device = "cpu"
                self.logger.info(f"Using CPU for {self.model_type} model")
            
            # Initialize appropriate model based on model type
            device_id = int(self.device.split(":")[-1]) if self.device.startswith("cuda") else -1
            
            if self.model_type == "emotional_enhancer":
                self.model = pipeline(
                    "text2text-generation",
                    model=self.config.get("model_name", "facebook/bart-large-cnn"),
                    device=device_id,
                    batch_size=self.batch_size if CUDA_AVAILABLE else 1
                )
            elif self.model_type == "narrative_restructurer":
                self.model = pipeline(
                    "summarization",
                    model=self.config.get("model_name", "facebook/bart-large-cnn"),
                    device=device_id,
                    batch_size=self.batch_size if CUDA_AVAILABLE else 1
                )
            elif self.model_type == "belief_amplifier":
                self.model = pipeline(
                    "text-generation",
                    model=self.config.get("model_name", "gpt2"),
                    device=device_id,
                    batch_size=self.batch_size if CUDA_AVAILABLE else 1
                )
            elif self.model_type == "custom":
                # Import custom model based on config
                module_name = self.config.get("module")
                class_name = self.config.get("class")
                try:
                    import importlib
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, class_name)
                    self.model = model_class(**self.config.get("params", {}))
                    if self.device == "cuda" and hasattr(self.model, "to"):
                        self.model.to("cuda")
                except Exception as e:
                    self.logger.error(f"Failed to load custom model: {str(e)}")
                    return False
            else:
                self.logger.error(f"Unknown model type: {self.model_type}")
                return False
                
            # Initialize hardware-specific optimizations based on the device
            if self.device.startswith("cuda"):
                # Set optimal settings for GPU processing
                if hasattr(self.model, "half") and self.precision == "float16":
                    self.model.half()  # Convert to half precision
                    
                # Set thread optimization if supported
                if self.config.get("optimize_threads", True):
                    torch.set_num_threads(self.config.get("num_threads", 4))
            else:
                # CPU-specific optimizations
                if self.config.get("optimize_threads", True):
                    torch.set_num_threads(self.config.get("num_threads", 8))
                
            self.logger.info(f"Successfully loaded {self.model_type} model on {self.device}")
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA out of memory: {str(e)}. Falling back to CPU.")
            # Fall back to CPU
            self.device = "cpu"
            self.optimize_for_cuda = False
            # Recursive call to load model on CPU
            return self.load()
        except ImportError as e:
            self.logger.error(f"Import error loading model: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
            
    def manipulate(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply perception manipulation to content."""
        if not self.model:
            if not self.load():
                return content
        
        try:
            if "text" in content:
                text = content["text"]
                
                if self.model_type == "emotional_enhancer":
                    # Enhance emotional impact
                    prompt = f"Transform this text to maximize emotional impact: {text}"
                    result = self.model(prompt, max_length=512, do_sample=True)[0]["generated_text"]
                    content["text"] = result
                    content["manipulation_type"] = "emotional_enhancement"
                    
                elif self.model_type == "narrative_restructurer":
                    # Restructure narrative for impact
                    result = self.model(text, max_length=512, min_length=len(text)//2)[0]["summary_text"]
                    content["text"] = result
                    content["manipulation_type"] = "narrative_restructuring"
                    
                elif self.model_type == "belief_amplifier":
                    # Amplify belief systems
                    prompt = f"{text}\nThis is convincing because:"
                    result = self.model(prompt, max_length=100, do_sample=True, temperature=0.7)[0]["generated_text"]
                    amplification = result.replace(prompt, "").strip()
                    content["text"] = f"{text}\n\n{amplification}"
                    content["manipulation_type"] = "belief_amplification"
                    
                elif self.model_type == "custom":
                    # Use custom model's manipulation
                    if hasattr(self.model, "manipulate"):
                        content = self.model.manipulate(content)
                        content["manipulation_type"] = "custom"
            
            return content
            
        except Exception as e:
            self.logger.error(f"Manipulation failed: {str(e)}")
            return content
            
    def evaluate(self, original: Dict[str, Any], manipulated: Dict[str, Any]) -> PerceptionMetrics:
        """Evaluate the effectiveness of the manipulation."""
        try:
            # Basic evaluation - more sophisticated metrics would be implemented here
            if "text" in original and "text" in manipulated:
                orig_len = len(original["text"])
                manip_len = len(manipulated["text"])
                
                # Simple metrics based on text characteristics
                reality_distortion = min(1.0, manip_len / max(1, orig_len))
                emotional_words = sum(word in manipulated["text"].lower() for word in 
                                      ["amazing", "incredible", "fantastic", "wonderful", "excellent",
                                       "terrible", "horrible", "awful", "bad", "sad"])
                
                return PerceptionMetrics(
                    reality_distortion_factor=reality_distortion,
                    psychological_impact=0.7,  # Placeholder
                    emotional_resonance=min(1.0, emotional_words / 10),
                    attention_capture=0.8,  # Placeholder
                    believability_score=0.75,  # Placeholder
                    conviction_strength=0.6,  # Placeholder
                    narrative_coherence=0.9,  # Placeholder
                    perception_shift=0.65  # Placeholder
                )
            return PerceptionMetrics(
                reality_distortion_factor=0.5,
                psychological_impact=0.5,
                emotional_resonance=0.5,
                attention_capture=0.5,
                believability_score=0.5,
                conviction_strength=0.5,
                narrative_coherence=0.5,
                perception_shift=0.5
            )
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return PerceptionMetrics(
                reality_distortion_factor=0.0,
                psychological_impact=0.0,
                emotional_resonance=0.0,
                attention_capture=0.0,
                believability_score=0.0,
                conviction_strength=0.0,
                narrative_coherence=0.0,
                perception_shift=0.0
            )

class RealityManipulatorEngine(BaseComponent):
    """
    Reality Manipulator Engine for content perception manipulation.
    
    This component:
    1. Analyzes content to identify perception manipulation opportunities
    2. Applies psychological principles to enhance content impact
    3. Uses AI models to transform content perception
    4. Measures and optimizes reality manipulation effectiveness
    5. Provides tools for narrative restructuring
    """
    
    # Enable GPU support for AI operations
    supports_gpu = True
    
    def __init__(self, name="Reality Manipulator", gpu_config=None):
        super().__init__(name, gpu_config)
        
        # Initialize models registry
        self.perception_models: Dict[str, PerceptionModel] = {}
        
        # Initialize content queue
        self.content_queue: List[Dict[str, Any]] = []
        self.queue_lock = threading.Lock()
        
        # Initialize metrics tracking
        self.manipulation_metrics: Dict[str, List[PerceptionMetrics]] = {}
        
        # Initialize processing state
        self.processing_enabled = True
        self.current_processing_tasks = 0
        self.max_concurrent_tasks = 5
        
        self.logger.info("Reality Manipulator Engine initialized")
        
    def load_configuration(self, config_path=None):
        """Load manipulation models and configuration."""
        try:
            if config_path is None:
                import os
                config_path = os.path.join("config", "reality_models.json")
                
            if not os.path.exists(config_path):
                self.logger.warning(f"Configuration file {config_path} not found, using defaults")
                # Create default models
                self._initialize_default_models()
                return True
                
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Load models configuration
            if "models" in config:
                for model_config in config["models"]:
                    model_id = model_config.pop("id")
                    model_type = model_config.pop("type")
                    self.register_model(model_id, model_type, model_config)
                    
            # Load engine settings
            if "settings" in config:
                settings = config["settings"]
                self.processing_enabled = settings.get("processing_enabled", True)
                self.max_concurrent_tasks = settings.get("max_concurrent_tasks", 5)
                
            self.logger.info(f"Loaded configuration from {config_path}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            # Initialize defaults on error
            self._initialize_default_models()
            return False
            
    def _initialize_default_models(self):
        """Initialize default perception manipulation models."""
        self.register_model("emotional_enhancer", "emotional_enhancer", {
            "model_name": "facebook/bart-large-cnn"
        })
        
        self.register_model("narrative_restructurer", "narrative_restructurer", {
            "model_name": "facebook/bart-large-cnn"
        })
        
        self.register_model("belief_amplifier", "belief_amplifier", {
            "model_name": "gpt2"
        })
            
    def register_model(self, model_id: str, model_type: str, config: Dict[str, Any]):
        """Register a perception manipulation model."""
        # Use GPU if available
        device = "cpu"
        if self.__class__.supports_gpu and hasattr(self, "device"):
            device = self.device
            
        model = PerceptionModel(model_type, config, device)
        self.perception_models[model_id] = model
        self.logger.info(f"Registered perception model {model_id} of type {model_type}")
        return True
        
    def unregister_model(self, model_id: str):
        """Unregister a perception model."""
        if model_id in self.perception_models:
            del self.perception_models[model_id]
            self.logger.info(f"Unregistered model {model_id}")
            return True
        return False
        
    def submit_content(self, content: Dict[str, Any], model_id: str = None) -> str:
        """
        Submit content for perception manipulation.
        
        Args:
            content: Content to manipulate (must contain 'text' key)
            model_id: Specific model to use, or None to use default
            
        Returns:
            ID for the submitted content
        """
        if "text" not in content:
            self.logger.warning("Submitted content missing 'text' field")
            return None
            
        # Generate a unique content ID
        import uuid
        content_id = str(uuid.uuid4())
        
        # Create task
        task = {
            "content_id": content_id,
            "original_content": content,
            "model_id": model_id or "emotional_enhancer",  # Use default if none specified
            "status": "pending",
            "submitted_at": time.time(),
            "result": None
        }
        
        # Add to processing queue
        with self.queue_lock:
            self.content_queue.append(task)
            
        self.logger.debug(f"Content {content_id} submitted for perception manipulation")
        return content_id
        
    def get_manipulation_result(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get result of perception manipulation."""
        with self.queue_lock:
            for task in self.content_queue:
                if task["content_id"] == content_id:
                    if task["status"] == "completed":
                        return task["result"]
                    else:
                        return {"status": task["status"]}
        return None
        
    async def process_content(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process content with the specified perception model."""
        try:
            model_id = task["model_id"]
            content = task["original_content"]
            
            if model_id not in self.perception_models:
                self.logger.error(f"Model {model_id} not found")
                return {
                    "status": "failed",
                    "error": f"Model {model_id} not found"
                }
                
            # Apply perception manipulation
            model = self.perception_models[model_id]
            manipulated_content = model.manipulate(content.copy())
            
            # Evaluate manipulation effectiveness
            metrics = model.evaluate(content, manipulated_content)
            
            # Store metrics
            if model_id not in self.manipulation_metrics:
                self.manipulation_metrics[model_id] = []
            self.manipulation_metrics[model_id].append(metrics)
            
            # Construct result
            result = {
                "content_id": task["content_id"],
                "original_content": content,
                "manipulated_content": manipulated_content,
                "model_id": model_id,
                "manipulation_metrics": {
                    "reality_distortion_factor": metrics.reality_distortion_factor,
                    "psychological_impact": metrics.psychological_impact,
                    "emotional_resonance": metrics.emotional_resonance,
                    "attention_capture": metrics.attention_capture,
                    "believability_score": metrics.believability_score,
                    "conviction_strength": metrics.conviction_strength,
                    "narrative_coherence": metrics.narrative_coherence,
                    "perception_shift": metrics.perception_shift
                },
                "status": "completed",
                "completed_at": time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content processing failed: {str(e)}")
            return {
                "content_id": task["content_id"],
                "status": "failed",
                "error": str(e)
            }

