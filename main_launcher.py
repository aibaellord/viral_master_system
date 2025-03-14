import os
import sys
import json
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ViralMasterSystem")

class SystemLauncher:
    def __init__(self):
        self.components = []
        self.load_config()
        
    def load_config(self):
        """Load system configuration."""
        try:
            with open("config/system_config.json", "r") as f:
                self.config = json.load(f)
            logger.info("System configuration loaded successfully")
            
            # Load GPU configuration if available
            try:
                with open("config/gpu_config.json", "r") as f:
                    self.gpu_config = json.load(f)
                logger.info(f"GPU configuration loaded: {self.gpu_config['use_gpu']}")
            except FileNotFoundError:
                logger.warning("GPU configuration not found. Running in CPU-only mode.")
                self.gpu_config = {"use_gpu": False}
                
        except FileNotFoundError:
            logger.error("System configuration not found. Creating default configuration.")
            self.create_default_config()
            
    def create_default_config(self):
        """Create default system configuration."""
        self.config = {
            "components": [
                {"name": "AI Orchestrator", "module": "core.engine.ai_orchestrator", "class": "AIOrchestratorEngine", "enabled": True},
                {"name": "Viral Orchestrator", "module": "core.engine.viral_orchestrator", "class": "ViralOrchestratorEngine", "enabled": True},
                {"name": "Analytics Engine", "module": "core.analytics.analytics_engine", "class": "AnalyticsEngine", "enabled": True},
                {"name": "Distribution Manager", "module": "core.distribution.distribution_manager", "class": "DistributionManager", "enabled": True}
            ],
            "system_settings": {
                "log_level": "INFO",
                "max_workers": 4,
                "auto_recovery": True
            }
        }
        
        os.makedirs("config", exist_ok=True)
        with open("config/system_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Default configuration created")
        
    def initialize_component(self, component_config):
        """Initialize a single system component."""
        try:
            if not component_config.get("enabled", True):
                logger.info(f"Component {component_config['name']} is disabled, skipping")
                return None
                
            logger.info(f"Initializing component: {component_config['name']}")
            module = importlib.import_module(component_config["module"])
            component_class = getattr(module, component_config["class"])
            
            # Pass GPU configuration if component supports it
            if hasattr(component_class, "supports_gpu") and component_class.supports_gpu:
                instance = component_class(gpu_config=self.gpu_config)
            else:
                instance = component_class()
                
            logger.info(f"Component {component_config['name']} initialized successfully")
            return instance
        except Exception as e:
            logger.error(f"Failed to initialize component {component_config['name']}: {str(e)}")
            return None
            
    def initialize_system(self):
        """Initialize all system components."""
        logger.info("Starting system initialization")
        
        # Initialize components with thread pool for parallel initialization
        with ThreadPoolExecutor(max_workers=self.config["system_settings"].get("max_workers", 4)) as executor:
            futures = [executor.submit(self.initialize_component, comp) for comp in self.config["components"]]
            
            for future in futures:
                component = future.result()
                if component:
                    self.components.append(component)
        
        logger.info(f"System initialization complete. {len(self.components)} components running.")
        
    def start_system(self):
        """Start all system components."""
        logger.info("Starting all system components")
        
        for component in self.components:
            if hasattr(component, "start"):
                try:
                    component.start()
                except Exception as e:
                    logger.error(f"Failed to start component {component.__class__.__name__}: {str(e)}")
        
        logger.info("All components started")
        
    def monitor_system(self):
        """Monitor system health and performance."""
        try:
            # Import monitoring modules dynamically
            monitoring = importlib.import_module("core.engine.monitoring_system")
            monitor = monitoring.MonitoringSystem(components=self.components)
            monitor.start_monitoring()
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {str(e)}")

if __name__ == "__main__":
    launcher = SystemLauncher()
    launcher.initialize_system()
    launcher.start_system()
    launcher.monitor_system()

