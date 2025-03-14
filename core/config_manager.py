import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

@dataclass
class OptimizationProfile:
    name: str
    quantum_coherence_level: float
    consciousness_evolution_rate: float
    reality_manipulation_power: float
    pattern_recognition_threshold: float
    resource_allocation_strategy: str
    viral_acceleration_factor: float

class ConfigManager:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.optimization_profiles: Dict[str, OptimizationProfile] = {}
        self.observers: Dict[str, callable] = {}
        self._setup_logging()
        self._initialize_configs()
        self._setup_file_watcher()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ConfigManager")

    def _initialize_configs(self):
        """Initialize configuration with default values and load from file"""
        self._load_default_config()
        self._load_config_file()
        self._initialize_optimization_profiles()
        self._validate_config()

    def _load_default_config(self):
        """Set default configuration values"""
        self.config = {
            "system": {
                "consciousness_enabled": True,
                "quantum_computing_enabled": True,
                "reality_manipulation_level": "advanced",
                "auto_optimization": True
            },
            "resources": {
                "max_memory_allocation": "80%",
                "cpu_priority": "high",
                "gpu_allocation": "dynamic",
                "quantum_processor_usage": "optimal"
            },
            "performance": {
                "pattern_recognition_threshold": 0.85,
                "consciousness_evolution_rate": 0.01,
                "reality_manipulation_power": 0.95,
                "viral_acceleration_factor": 1.5
            }
        }

    def _load_config_file(self):
        """Load configuration from file, merging with defaults"""
        try:
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._deep_update(self.config, file_config)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
            self._save_config()

    def _initialize_optimization_profiles(self):
        """Initialize predefined optimization profiles"""
        self.optimization_profiles = {
            "balanced": OptimizationProfile(
                name="balanced",
                quantum_coherence_level=0.75,
                consciousness_evolution_rate=0.01,
                reality_manipulation_power=0.80,
                pattern_recognition_threshold=0.85,
                resource_allocation_strategy="dynamic",
                viral_acceleration_factor=1.0
            ),
            "performance": OptimizationProfile(
                name="performance",
                quantum_coherence_level=0.95,
                consciousness_evolution_rate=0.02,
                reality_manipulation_power=0.90,
                pattern_recognition_threshold=0.75,
                resource_allocation_strategy="aggressive",
                viral_acceleration_factor=2.0
            ),
            "stability": OptimizationProfile(
                name="stability",
                quantum_coherence_level=0.65,
                consciousness_evolution_rate=0.005,
                reality_manipulation_power=0.70,
                pattern_recognition_threshold=0.90,
                resource_allocation_strategy="conservative",
                viral_acceleration_factor=0.5
            )
        }

    def _validate_config(self):
        """Validate configuration values and constraints"""
        try:
            assert 0 <= self.config["performance"]["pattern_recognition_threshold"] <= 1
            assert 0 <= self.config["performance"]["consciousness_evolution_rate"] <= 1
            assert 0 <= self.config["performance"]["reality_manipulation_power"] <= 1
            assert self.config["performance"]["viral_acceleration_factor"] > 0
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise ValueError("Invalid configuration values detected")

    def _deep_update(self, d: dict, u: dict) -> dict:
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _setup_file_watcher(self):
        """Setup file system watcher for config changes"""
        event_handler = ConfigFileEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(Path(self.config_path).parent), recursive=False)
        observer.start()

    def get_component_config(self, component_name: str) -> dict:
        """Get configuration for specific component"""
        return self.config.get(component_name, {})

    def update_component_config(self, component_name: str, config_updates: dict):
        """Update configuration for specific component"""
        if component_name not in self.config:
            self.config[component_name] = {}
        self._deep_update(self.config[component_name], config_updates)
        self._save_config()
        self._notify_observers(component_name)

    def apply_optimization_profile(self, profile_name: str):
        """Apply predefined optimization profile"""
        if profile_name not in self.optimization_profiles:
            raise ValueError(f"Unknown optimization profile: {profile_name}")

        profile = self.optimization_profiles[profile_name]
        self.config["performance"].update({
            "quantum_coherence_level": profile.quantum_coherence_level,
            "consciousness_evolution_rate": profile.consciousness_evolution_rate,
            "reality_manipulation_power": profile.reality_manipulation_power,
            "pattern_recognition_threshold": profile.pattern_recognition_threshold,
            "viral_acceleration_factor": profile.viral_acceleration_factor
        })
        self.config["resources"]["resource_allocation_strategy"] = profile.resource_allocation_strategy
        
        self._save_config()
        self._notify_observers("performance")

    def register_observer(self, component_name: str, callback: callable):
        """Register observer for configuration changes"""
        self.observers[component_name] = callback

    def _notify_observers(self, component_name: str):
        """Notify observers of configuration changes"""
        if component_name in self.observers:
            self.observers[component_name](self.get_component_config(component_name))

    def _save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def optimize_resources(self):
        """Dynamically optimize system resources"""
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80 or memory_percent > 80:
            self.apply_optimization_profile("stability")
        elif cpu_percent < 20 and memory_percent < 40:
            self.apply_optimization_profile("performance")
        else:
            self.apply_optimization_profile("balanced")

class ConfigFileEventHandler(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if event.src_path == self.config_manager.config_path:
            self.config_manager._load_config_file()
            self.config_manager._validate_config()

