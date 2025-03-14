from __future__ import annotations
import os
import json
import yaml
import redis
import logging
import threading
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, ValidationError
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class ConfigVersion:
    version: str
    timestamp: datetime
    changes: Dict[str, Any]
    author: str

class ConfigValidationSchema(BaseModel):
    environment: str
    debug_mode: bool
    log_level: str
    redis_host: str
    redis_port: int
    secret_key: str
    max_retries: int
    timeout: int
    
class ConfigManager:
    def __init__(self, base_path: str = "configs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.config_lock = threading.Lock()
        self.current_config: Dict[str, Any] = {}
        self.version_history: List[ConfigVersion] = []
        self._setup_encryption()
        self._setup_redis()
        self._load_config()
        self._start_monitoring()

    def _setup_encryption(self) -> None:
        """Initialize encryption for secure secret management"""
        key = os.environ.get("CONFIG_ENCRYPTION_KEY")
        if not key:
            key = Fernet.generate_key()
            os.environ["CONFIG_ENCRYPTION_KEY"] = key.decode()
        self.cipher_suite = Fernet(key if isinstance(key, bytes) else key.encode())

    def _setup_redis(self) -> None:
        """Initialize Redis for distributed configuration"""
        self.redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True
        )

    def _load_config(self) -> None:
        """Load configuration from files with environment overrides"""
        with self.config_lock:
            base_config = self._load_yaml("base_config.yaml")
            env = os.environ.get("ENVIRONMENT", "development")
            env_config = self._load_yaml(f"{env}_config.yaml")
            self.current_config = self._merge_configs(base_config, env_config)
            self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration against schema"""
        try:
            ConfigValidationSchema(**self.current_config)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split(".")
        value = self.current_config
        for k in keys:
            value = value.get(k, default)
            if value == default:
                return default
        return value

    def set(self, key: str, value: Any, author: str) -> None:
        """Update configuration value with versioning"""
        with self.config_lock:
            old_value = self.get(key)
            if old_value == value:
                return
            
            keys = key.split(".")
            config = self.current_config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            
            version = ConfigVersion(
                version=f"v{len(self.version_history) + 1}",
                timestamp=datetime.utcnow(),
                changes={key: {"old": old_value, "new": value}},
                author=author
            )
            self.version_history.append(version)
            self._validate_config()
            self._publish_update(key, value)

    def _publish_update(self, key: str, value: Any) -> None:
        """Publish configuration updates to Redis"""
        try:
            self.redis_client.publish(
                "config_updates",
                json.dumps({"key": key, "value": value})
            )
        except redis.RedisError as e:
            logger.error(f"Failed to publish config update: {e}")

    def backup(self, backup_path: Optional[str] = None) -> None:
        """Create configuration backup"""
        if not backup_path:
            backup_path = self.base_path / f"backup_{datetime.utcnow().isoformat()}.yaml"
        
        with open(backup_path, "w") as f:
            yaml.safe_dump({
                "config": self.current_config,
                "versions": [asdict(v) for v in self.version_history]
            }, f)

    def restore(self, backup_path: str) -> None:
        """Restore configuration from backup"""
        with open(backup_path, "r") as f:
            data = yaml.safe_load(f)
            with self.config_lock:
                self.current_config = data["config"]
                self.version_history = [
                    ConfigVersion(**v) for v in data["versions"]
                ]
                self._validate_config()

    def get_optimization_suggestions(self) -> List[str]:
        """Generate configuration optimization suggestions"""
        suggestions = []
        
        # Example suggestions based on patterns
        if self.get("debug_mode", False):
            suggestions.append("Debug mode is enabled in production")
        
        if self.get("max_retries", 0) < 3:
            suggestions.append("Consider increasing max_retries for better reliability")
        
        return suggestions

    def _start_monitoring(self) -> None:
        """Start real-time configuration monitoring"""
        def monitor():
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("config_updates")
            
            for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    if data["key"] not in self.current_config:
                        logger.info(f"Received update for new config key: {data['key']}")
                    
        threading.Thread(target=monitor, daemon=True).start()

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge configuration dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def _load_yaml(self, filename: str) -> Dict:
        """Load and parse YAML configuration file"""
        try:
            config_file = self.base_path / filename
            if config_file.exists():
                with open(config_file, "r") as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config file {filename}: {e}")
        return {}

