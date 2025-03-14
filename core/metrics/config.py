from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import yaml
from enum import Enum
import re

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""
    pass

class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"

class CacheType(Enum):
    REDIS = "redis"
    MEMCACHED = "memcached"
    IN_MEMORY = "in_memory"

@dataclass
class DatabaseConfig:
    type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = "metrics"
    username: str = "metrics_user"
    password: str = field(default="", repr=False)
    pool_size: int = 5
    max_overflow: int = 10
    timeout: int = 30

    def validate(self) -> None:
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ConfigValidationError(f"Invalid database port: {self.port}")
        if not isinstance(self.pool_size, int) or self.pool_size < 1:
            raise ConfigValidationError(f"Invalid pool size: {self.pool_size}")
        if not isinstance(self.max_overflow, int) or self.max_overflow < 0:
            raise ConfigValidationError(f"Invalid max overflow: {self.max_overflow}")
        if not isinstance(self.timeout, int) or self.timeout < 1:
            raise ConfigValidationError(f"Invalid timeout: {self.timeout}")

@dataclass
class CacheConfig:
    type: CacheType
    host: str = "localhost"
    port: int = 6379
    ttl: int = 3600
    max_size: int = 1000
    eviction_policy: str = "lru"

    def validate(self) -> None:
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ConfigValidationError(f"Invalid cache port: {self.port}")
        if not isinstance(self.ttl, int) or self.ttl < 0:
            raise ConfigValidationError(f"Invalid TTL: {self.ttl}")
        if not isinstance(self.max_size, int) or self.max_size < 1:
            raise ConfigValidationError(f"Invalid max size: {self.max_size}")
        if self.eviction_policy not in ["lru", "lfu", "fifo"]:
            raise ConfigValidationError(f"Invalid eviction policy: {self.eviction_policy}")

@dataclass
class PlatformConfig:
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    rate_limit: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5

    def validate(self) -> None:
        if not isinstance(self.rate_limit, int) or self.rate_limit < 1:
            raise ConfigValidationError(f"Invalid rate limit: {self.rate_limit}")
        if not isinstance(self.timeout, int) or self.timeout < 1:
            raise ConfigValidationError(f"Invalid timeout: {self.timeout}")
        if not isinstance(self.retry_attempts, int) or self.retry_attempts < 0:
            raise ConfigValidationError(f"Invalid retry attempts: {self.retry_attempts}")
        if not isinstance(self.retry_delay, int) or self.retry_delay < 1:
            raise ConfigValidationError(f"Invalid retry delay: {self.retry_delay}")

@dataclass
class AlertConfig:
    enabled: bool = True
    thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=lambda: ["email"])
    cooldown_period: int = 300

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigValidationError("Alert enabled must be a boolean")
        if not isinstance(self.thresholds, dict):
            raise ConfigValidationError("Thresholds must be a dictionary")
        if not isinstance(self.notification_channels, list):
            raise ConfigValidationError("Notification channels must be a list")
        if not isinstance(self.cooldown_period, int) or self.cooldown_period < 0:
            raise ConfigValidationError(f"Invalid cooldown period: {self.cooldown_period}")

@dataclass
class RetentionConfig:
    raw_data_days: int = 30
    aggregated_data_days: int = 365
    backup_enabled: bool = True
    backup_interval_hours: int = 24

    def validate(self) -> None:
        if not isinstance(self.raw_data_days, int) or self.raw_data_days < 1:
            raise ConfigValidationError(f"Invalid raw data retention: {self.raw_data_days}")
        if not isinstance(self.aggregated_data_days, int) or self.aggregated_data_days < 1:
            raise ConfigValidationError(f"Invalid aggregated data retention: {self.aggregated_data_days}")
        if not isinstance(self.backup_interval_hours, int) or self.backup_interval_hours < 1:
            raise ConfigValidationError(f"Invalid backup interval: {self.backup_interval_hours}")

@dataclass
class ExportConfig:
    formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    compression: bool = True
    max_batch_size: int = 1000
    include_metadata: bool = True

    def validate(self) -> None:
        valid_formats = ["json", "csv", "parquet", "excel"]
        if not all(fmt in valid_formats for fmt in self.formats):
            raise ConfigValidationError(f"Invalid export formats: {self.formats}")
        if not isinstance(self.max_batch_size, int) or self.max_batch_size < 1:
            raise ConfigValidationError(f"Invalid max batch size: {self.max_batch_size}")

class MetricsConfig:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.platform = PlatformConfig()
        self.alerts = AlertConfig()
        self.retention = RetentionConfig()
        self.export = ExportConfig()
        if config_path:
            self.load()
        self._apply_env_vars()

    def load(self, config_path: Optional[Path] = None) -> None:
        """Load configuration from YAML file."""
        if config_path:
            self.config_path = config_path
        if not self.config_path:
            raise ConfigLoadError("No configuration path specified")
        
        try:
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)
            
            if not isinstance(config_data, dict):
                raise ConfigLoadError("Invalid configuration format")
            
            # Load each section with inheritance support
            if "database" in config_data:
                self.database = DatabaseConfig(**config_data["database"])
            if "cache" in config_data:
                self.cache = CacheConfig(**config_data["cache"])
            if "platform" in config_data:
                self.platform = PlatformConfig(**config_data["platform"])
            if "alerts" in config_data:
                self.alerts = AlertConfig(**config_data["alerts"])
            if "retention" in config_data:
                self.retention = RetentionConfig(**config_data["retention"])
            if "export" in config_data:
                self.export = ExportConfig(**config_data["export"])
            
            self.validate()
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load configuration: {e}")

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if config_path:
            self.config_path = config_path
        if not self.config_path:
            raise ConfigLoadError("No configuration path specified")
        
        config_data = {
            "database": {k: v for k, v in self.database.__dict__.items()},
            "cache": {k: v for k, v in self.cache.__dict__.items()},
            "platform": {k: v for k, v in self.platform.__dict__.items()},
            "alerts": {k: v for k, v in self.alerts.__dict__.items()},
            "retention": {k: v for k, v in self.retention.__dict__.items()},
            "export": {k: v for k, v in self.export.__dict__.items()},
        }
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
        except Exception as e:
            raise ConfigLoadError(f"Failed to save configuration: {e}")

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.database.validate()
        self.cache.validate()
        self.platform.validate()
        self.alerts.validate()
        self.retention.validate()
        self.export.validate()

    def _apply_env_vars(self) -> None:
        """Apply environment variable overrides."""
        env_mapping = {
            "METRICS_DB_HOST": ("database", "host"),
            "METRICS_DB_PORT": ("database", "port", int),
            "METRICS_DB_NAME": ("database", "database"),
            "METRICS_DB_USER": ("database", "username"),
            "METRICS_DB_PASS": ("database", "password"),
            "METRICS_CACHE_HOST": ("cache", "host"),
            "METRICS_CACHE_PORT": ("cache", "port", int),
            "METRICS_CACHE_TTL": ("cache", "ttl", int),
            "METRICS_PLATFORM_KEY": ("platform", "api_key"),
            "METRICS_PLATFORM_SECRET": ("platform", "api_secret"),
            "METRICS_ALERT_ENABLED": ("alerts", "enabled", lambda x: x.lower() == "true"),
        }

        for env_var, (section, key, *converter) in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if converter:
                    value = converter[0](value)
                setattr(getattr(self, section), key, value)

    def export_config(self, format: str = "yaml") -> Union[str, dict]:
        """Export configuration in the specified format."""
        config_data = {
            "database": {k: v for k, v in self.database.__dict__.items() 
                        if not k.startswith("_")},
            "cache": {k: v for k, v in self.cache.__dict__.items() 
                     if not k.startswith("_")},
            "platform": {k: v for k, v in self.platform.__dict__.items() 
                        if not k.startswith("_")},
            "alerts": {k: v for k, v in self.alerts.__dict__.items() 
                      if not k.startswith("_")},
            "retention": {k: v for k, v in self.retention.__dict__.items() 
                         if not k.startswith("_")},
            "export": {k: v for k, v in self.export.__dict__.items() 
                      if not k.startswith("_")},
        }
        
        if format == "yaml":
            return yaml.safe_dump(config_data, default_flow_style=False)
        elif format == "dict":
            return config_data
        else:
            raise ValueError(f"Unsupported export format: {format}")

