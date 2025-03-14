import logging
import threading
import time

class BaseComponent:
    """Base class for all system components with common functionality."""
    
    supports_gpu = False  # Override in GPU-compatible components
    
    def __init__(self, name=None, gpu_config=None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"ViralSystem.{self.name}")
        self.running = False
        self.thread = None
        self.gpu_config = gpu_config
        
        # Initialize GPU if supported and available
        if self.__class__.supports_gpu and gpu_config and gpu_config.get("use_gpu", False):
            self.initialize_gpu()
        
    def initialize_gpu(self):
        """Initialize GPU acceleration for this component."""
        self.logger.info(f"Initializing GPU support for {self.name}")
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                # Apply RTX-specific optimizations if enabled
                if self.gpu_config.get("rtx_specific_optimizations", False):
                    torch.backends.cudnn.benchmark = True
                    if torch.__version__ >= "1.7.0":
                        # Use AMP (Automatic Mixed Precision) for RTX cards
                        self.scaler = torch.cuda.amp.GradScaler()
                        self.logger.info("RTX optimizations enabled: Mixed precision training")
            else:
                self.logger.warning("GPU requested but not available. Falling back to CPU.")
        except ImportError:
            self.logger.error("PyTorch not installed. Cannot use GPU acceleration.")
            self.device = "cpu"
        except Exception as e:
            self.logger.error(f"Error initializing GPU: {str(e)}")
            self.device = "cpu"
            
    def start(self):
        """Start the component in a separate thread."""
        if self.running:
            self.logger.warning(f"{self.name} is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"{self.name} started")
        
    def stop(self):
        """Stop the component."""
        self.logger.info(f"Stopping {self.name}")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info(f"{self.name} stopped")
        
    def run(self):
        """Main execution method. Override in derived classes."""
        self.logger.warning(f"Default run method called in {self.name}. Should be overridden.")
        while self.running:
            time.sleep(1)
            
    def get_status(self):
        """Return component status."""
        return {
            "name": self.name,
            "running": self.running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "using_gpu": hasattr(self, "device") and getattr(self, "device", "cpu") != "cpu"
        }

