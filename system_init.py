#!/usr/bin/env python3
"""
Unified System Initialization Script for Hyper-Automated Viral Master System

This script provides a single entry point to start the entire system,
handling dependency checks, configuration, component initialization,
and startup of all required services.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import importlib
import subprocess
from pathlib import Path

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ViralSystem.Initializer")

class SystemInitializer:
    """
    Main class responsible for initializing and starting all system components.
    """
    def __init__(self, args):
        self.args = args
        self.launcher = None
        self.dashboard_thread = None
        self.config_dir = Path("config")
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.config_dir,
            Path("templates"),
            Path("static"),
            Path("logs")
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            
    def check_dependencies(self):
        """
        Check if all required dependencies are installed.
        """
        logger.info("Checking system dependencies...")
        
        # Check Python version
        py_version = sys.version_info
        if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 6):
            logger.error("Python 3.6 or higher is required")
            return False
            
        # Check required Python packages
        required_packages = [
            "numpy", "pandas", "flask", "requests", 
            "matplotlib", "torch", "transformers"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            logger.warning(f"Missing Python packages: {', '.join(missing_packages)}")
            if self.args.auto_install:
                self.install_dependencies(missing_packages)
            else:
                logger.error("Please run ./install_dependencies.sh to install required packages")
                return False
                
        logger.info("All required dependencies are installed")
        return True
        
    def install_dependencies(self, packages=None):
        """
        Install missing dependencies if auto-install is enabled.
        """
        logger.info("Installing dependencies...")
        
        if os.path.exists("./install_dependencies.sh"):
            try:
                if packages:
                    # Install specific packages
                    subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
                else:
                    # Run the full installation script
                    subprocess.run(["./install_dependencies.sh"], check=True)
                logger.info("Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {str(e)}")
                return False
        else:
            logger.error("install_dependencies.sh not found")
            return False
            
    def configure_gpu(self):
        """
        Configure GPU if available and setup_gpu.py exists.
        """
        logger.info("Checking GPU configuration...")
        
        if os.path.exists("setup_gpu.py"):
            try:
                # Run the GPU configuration script
                subprocess.run([sys.executable, "setup_gpu.py"], check=True)
                logger.info("GPU configuration completed")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"GPU configuration failed: {str(e)}")
                # Continue with CPU mode
                return False
        else:
            logger.warning("setup_gpu.py not found, skipping GPU configuration")
            # Create a default CPU configuration
            gpu_config = {
                "has_cuda": False,
                "gpu_count": 0,
                "use_gpu": False
            }
            with open(self.config_dir / "gpu_config.json", "w") as f:
                json.dump(gpu_config, f, indent=2)
            logger.info("Created default CPU configuration")
            return True
            
    def initialize_system(self):
        """
        Initialize the system launcher and all components.
        """
        logger.info("Initializing system...")
        
        # Import the main launcher
        try:
            launcher_module = importlib.import_module("main_launcher")
            self.launcher = launcher_module.SystemLauncher()
            self.launcher.initialize_system()
            logger.info("System initialized successfully")
            return True
        except ImportError:
            logger.error("Failed to import main_launcher module")
            return False
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False
            
    def start_dashboard(self):
        """
        Start the web dashboard in a separate thread.
        """
        if not self.args.no_dashboard:
            logger.info("Starting web dashboard...")
            
            def run_dashboard():
                try:
                    dashboard_module = importlib.import_module("dashboard")
                    dashboard_module.start_dashboard(self.launcher)
                    dashboard_module.app.run(
                        host=self.args.host,
                        port=self.args.port,
                        debug=False
                    )
                except ImportError:
                    logger.error("Failed to import dashboard module")
                except Exception as e:
                    logger.error(f"Dashboard startup failed: {str(e)}")
                    
            self.dashboard_thread = threading.Thread(target=run_dashboard)
            self.dashboard_thread.daemon = True
            self.dashboard_thread.start()
            logger.info(f"Dashboard started at http://{self.args.host}:{self.args.port}")
            
    def start_system(self):
        """
        Start the system and all its components.
        """
        logger.info("Starting the system...")
        
        if self.launcher:
            try:
                self.launcher.start_system()
                
                # Start monitoring if enabled
                if not self.args.no_monitor:
                    self.launcher.monitor_system()
                    
                logger.info("System started successfully")
                
                # Keep the main thread alive
                self.keep_alive()
                
                return True
            except Exception as e:
                logger.error(f"System startup failed: {str(e)}")
                return False
        else:
            logger.error("Cannot start system: Launcher not initialized")
            return False
            
    def keep_alive(self):
        """
        Keep the main thread alive and handle graceful shutdown.
        """
        try:
            logger.info("System is now running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            self.shutdown()
            
    def shutdown(self):
        """
        Perform a graceful shutdown of all components.
        """
        logger.info("Shutting down the system...")
        
        if self.launcher:
            # Stop all components
            for component in self.launcher.components:
                if hasattr(component, 'stop'):
                    try:
                        component.stop()
                    except Exception as e:
                        logger.error(f"Error stopping component {component.name}: {str(e)}")
                        
        logger.info("System shutdown complete")
        
    def run(self):
        """
        Main method to run the entire initialization and startup sequence.
        """
        # Steps to initialize and start the system
        if not self.check_dependencies():
            logger.error("Dependency check failed, aborting startup")
            return False
            
        if not self.configure_gpu():
            logger.warning("GPU configuration failed, continuing with CPU mode")
            
        if not self.initialize_system():
            logger.error("System initialization failed, aborting startup")
            return False
            
        self.start_dashboard()
        
        return self.start_system()


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Unified System Initialization for Hyper-Automated Viral Master System"
    )
    
    parser.add_argument(
        "--auto-install", 
        action="store_true",
        help="Automatically install missing dependencies"
    )
    
    parser.add_argument(
        "--no-dashboard", 
        action="store_true",
        help="Disable the web dashboard"
    )
    
    parser.add_argument(
        "--no-monitor", 
        action="store_true",
        help="Disable system monitoring"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host for the web dashboard (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port for the web dashboard (default: 5000)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Custom configuration file path"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create and run the system initializer
    initializer = SystemInitializer(args)
    success = initializer.run()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

