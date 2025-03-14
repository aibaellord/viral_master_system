"""
System Orchestrator for Content Optimization Platform

This module serves as the central coordination hub for all system components,
managing workflows, data flow, configuration, API endpoints, error handling,
and optimization processes.

It integrates ContentPerformanceOptimizer, NeuralContentEnhancer, and RealTimeMonitor
into a cohesive system with standardized interfaces and workflows.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

# Import system components
# Note: These imports should be adjusted based on your actual file structure
try:
    from core.engine.content_performance_optimizer import ContentPerformanceOptimizer
    from core.engine.neural_content_enhancer import NeuralContentEnhancer
    from core.engine.real_time_monitor import RealTimeMonitor
except ImportError:
    # Fallback imports for testing or direct usage
    from content_performance_optimizer import ContentPerformanceOptimizer
    from neural_content_enhancer import NeuralContentEnhancer
    from real_time_monitor import RealTimeMonitor


# Configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("SystemOrchestrator")


class ProcessingState(Enum):
    """Enumeration for tracking the state of content processing."""
    INITIATED = "initiated"
    ANALYZING = "analyzing"
    ENHANCING = "enhancing"
    OPTIMIZING = "optimizing"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingContext:
    """Context object for tracking content processing throughout the system."""
    content_id: str
    state: ProcessingState
    start_time: float
    last_updated: float
    platform: str
    content_type: str
    metrics: Dict[str, Any] = None
    error: Optional[str] = None
    processing_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.processing_history is None:
            self.processing_history = []
    
    def update_state(self, new_state: ProcessingState, metrics: Dict[str, Any] = None) -> None:
        """Update the processing state and record the transition in history."""
        previous_state = self.state
        self.state = new_state
        self.last_updated = time.time()
        
        if metrics:
            self.metrics.update(metrics)
        
        # Record the state transition
        self.processing_history.append({
            "timestamp": self.last_updated,
            "from_state": previous_state.value,
            "to_state": new_state.value,
            "metrics": metrics
        })
    
    def add_error(self, error_message: str) -> None:
        """Add error information to the context."""
        self.error = error_message
        self.update_state(ProcessingState.FAILED)
    
    def get_processing_time(self) -> float:
        """Calculate the total processing time."""
        return self.last_updated - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary for serialization."""
        return {
            "content_id": self.content_id,
            "state": self.state.value,
            "start_time": self.start_time,
            "last_updated": self.last_updated,
            "platform": self.platform,
            "content_type": self.content_type,
            "metrics": self.metrics,
            "error": self.error,
            "processing_history": self.processing_history,
            "total_processing_time": self.get_processing_time()
        }


class SystemOrchestrator:
    """
    Main system orchestrator that coordinates interactions between all components 
    of the content optimization platform.
    
    Responsibilities:
    - Coordinate component interactions
    - Manage data flow through processing pipelines
    - Handle system-wide configuration
    - Provide unified API endpoints
    - Implement error handling and recovery
    - Manage optimization workflows
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SystemOrchestrator with configuration settings.
        
        Args:
            config_path: Path to the configuration file (JSON)
        """
        self.logger = logging.getLogger("SystemOrchestrator")
        self.logger.info("Initializing System Orchestrator")
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize component tracker
        self.active_components = {}
        self.component_health = {}
        
        # Initialize processing context registry
        self.processing_contexts = {}
        
        # Initialize components with appropriate configuration
        self._init_components()
        
        # Set up threading resources
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 10))
        
        # Set up event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Setup error handling and recovery
        self._setup_error_handling()
        
        self.logger.info("System Orchestrator initialized successfully")
    
    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from a file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration values
        """
        default_config = {
            "max_workers": 10,
            "retry_attempts": 3,
            "retry_delay": 2,  # seconds
            "monitoring_interval": 30,  # seconds
            "optimization_threshold": 0.7,
            "enable_real_time_monitoring": True,
            "log_level": "INFO",
            "component_settings": {
                "content_performance_optimizer": {
                    "enabled": True,
                    "model_path": "./models/optimizer",
                    "training_frequency": 86400  # daily in seconds
                },
                "neural_content_enhancer": {
                    "enabled": True,
                    "model_path": "./models/enhancer",
                    "enhancement_threshold": 0.6
                },
                "real_time_monitor": {
                    "enabled": True,
                    "alert_threshold": 0.3,
                    "check_interval": 60  # seconds
                }
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults for any missing values
                    for key, value in loaded_config.items():
                        if key == "component_settings" and key in default_config:
                            for component, settings in value.items():
                                if component in default_config["component_settings"]:
                                    default_config["component_settings"][component].update(settings)
                                else:
                                    default_config["component_settings"][component] = settings
                        else:
                            default_config[key] = value
                    
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
                self.logger.warning("Using default configuration")
        
        # Set log level from config
        log_level = getattr(logging, default_config.get("log_level", "INFO"))
        logging.getLogger().setLevel(log_level)
        
        return default_config
    
    def _init_components(self) -> None:
        """Initialize all system components with appropriate configuration."""
        try:
            # Initialize ContentPerformanceOptimizer
            optimizer_config = self.config["component_settings"]["content_performance_optimizer"]
            if optimizer_config.get("enabled", True):
                self.logger.info("Initializing ContentPerformanceOptimizer")
                self.optimizer = ContentPerformanceOptimizer(
                    model_path=optimizer_config.get("model_path", "./models/optimizer"),
                    training_frequency=optimizer_config.get("training_frequency", 86400)
                )
                self.active_components["optimizer"] = self.optimizer
                self.component_health["optimizer"] = True
            
            # Initialize NeuralContentEnhancer
            enhancer_config = self.config["component_settings"]["neural_content_enhancer"]
            if enhancer_config.get("enabled", True):
                self.logger.info("Initializing NeuralContentEnhancer")
                self.enhancer = NeuralContentEnhancer(
                    model_path=enhancer_config.get("model_path", "./models/enhancer"),
                    enhancement_threshold=enhancer_config.get("enhancement_threshold", 0.6)
                )
                self.active_components["enhancer"] = self.enhancer
                self.component_health["enhancer"] = True
            
            # Initialize RealTimeMonitor
            monitor_config = self.config["component_settings"]["real_time_monitor"]
            if monitor_config.get("enabled", True):
                self.logger.info("Initializing RealTimeMonitor")
                self.monitor = RealTimeMonitor(
                    alert_threshold=monitor_config.get("alert_threshold", 0.3),
                    check_interval=monitor_config.get("check_interval", 60)
                )
                self.active_components["monitor"] = self.monitor
                self.component_health["monitor"] = True
            
            self.logger.info(f"Components initialized: {list(self.active_components.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Component initialization failed: {str(e)}")
    
    def _setup_error_handling(self) -> None:
        """Set up error handling and recovery mechanisms."""
        # Set up exception handlers for threads
        threading.excepthook = self._handle_thread_exception
        
        # Start health monitoring
        if self.config.get("enable_real_time_monitoring", True):
            self.health_monitor_thread = threading.Thread(
                target=self._monitor_component_health, 
                daemon=True
            )
            self.health_monitor_thread.start()
            self.logger.info("Component health monitoring started")
    
    def _handle_thread_exception(self, args) -> None:
        """Custom exception handler for thread exceptions."""
        self.logger.error(f"Unhandled exception in thread: {args.exc_value}")
        self.logger.error(f"Thread: {args.thread}")
        self.logger.error(traceback.format_tb(args.exc_traceback))
    
    def _monitor_component_health(self) -> None:
        """Continuously monitor the health of all components."""
        while True:
            try:
                for component_name, component in self.active_components.items():
                    try:
                        # Check if component has a health check method
                        if hasattr(component, "check_health") and callable(component.check_health):
                            is_healthy = component.check_health()
                            
                            # Record previous state to detect changes
                            previous_state = self.component_health.get(component_name, False)
                            self.component_health[component_name] = is_healthy
                            
                            # Log state changes
                            if previous_state != is_healthy:
                                if is_healthy:
                                    self.logger.info(f"Component {component_name} is now healthy")
                                else:
                                    self.logger.warning(f"Component {component_name} is unhealthy")
                                    
                                    # Attempt recovery
                                    self._attempt_component_recovery(component_name, component)
                    except Exception as e:
                        self.logger.error(f"Error checking health of {component_name}: {str(e)}")
                        self.component_health[component_name] = False
                        
                        # Attempt recovery on exception
                        self._attempt_component_recovery(component_name, component)
                
                # Check overall system health
                self._check_system_health()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get("monitoring_interval", 30))
            except Exception as e:
                self.logger.error(f"Error in health monitoring thread: {str(e)}")
                time.sleep(self.config.get("monitoring_interval", 30))
    
    def _attempt_component_recovery(self, component_name: str, component: Any) -> bool:
        """
        Attempt to recover a failed component.
        
        Args:
            component_name: Name of the component
            component: Component instance
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        self.logger.info(f"Attempting to recover component: {component_name}")
        
        try:
            # Check if component has a reset method
            if hasattr(component, "reset") and callable(component.reset):
                component.reset()
                self.logger.info(f"Reset component: {component_name}")
                
                # Verify health after reset
                if hasattr(component, "check_health") and callable(component.check_health):
                    is_healthy = component.check_health()
                    self.component_health[component_name] = is_healthy
                    
                    if is_healthy:
                        self.logger.info(f"Component {component_name} recovered successfully")
                        return True
            
            # If no reset method or reset didn't help, try to reinitialize
            component_config = self.config["component_settings"].get(component_name, {})
            component_class = component.__class__
            
            # Reinitialize component with its configuration
            new_component = component_class(**component_config)
            self.active_components[component_name] = new_component
            
            # Check health of reinitialized component
            if hasattr(new_component, "check_health") and callable(new_component.check_health):
                is_healthy = new_component.check_health()
                self.component_health[component_name] = is_healthy
                
                if is_healthy:
                    self.logger.info(f"Component {component_name} reinitialized successfully")
                    return True
            
            self.logger.warning(f"Failed to recover component: {component_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error recovering component {component_name}: {str(e)}")
            self.component_health[component_name] = False
            return False
    
    def _check_system_health(self) -> bool:
        """
        Check overall system health based on component statuses.
        
        Returns:
            bool: True if system is healthy, False otherwise
        """
        if not self.component_health:
            self.logger.warning("No components are being tracked for health")
            return False
        
        # Count healthy components
        healthy_count = sum(1 for healthy in self.component_health.values() if healthy)
        total_count = len(self.component_health)
        
        # Calculate health percentage
        health_percentage = healthy_count

