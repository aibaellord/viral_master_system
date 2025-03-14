import os
import pytest
from pathlib import Path
from datetime import timedelta
import yaml

from core.metrics.config import (
    ConfigurationManager,
    MetricsConfig,
    DatabaseConfig,
    CacheConfig,
    PlatformConfig,
    AlertThresholds,
    RetentionPolicy,
    PerformanceSettings,
    ExportConfig,
)

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary configuration file for testing."""
    config_path = tmp_path / "test_config.yml"
    config_content = {
        "database": {
            "host": "testdb.example.com",
            "port": 5432,
            "database": "test_metrics",
        },
        "cache": {
            "backend": "redis",
            "host": "cache.example.com",
            "ttl": 7200,
        },
        "platforms": {
            "instagram": {
                "enabled": True,
                "api_key": "test_key",
                "rate_limit": 200,
            }
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_content, f)
    return config_path

def test_default_config():
    """Test configuration defaults are set correctly."""
    config = MetricsConfig()
    
    assert isinstance(config.database, DatabaseConfig)
    assert config.database.host == "localhost"
    assert config.database.port == 5432
    
    assert isinstance(config.cache, CacheConfig)
    assert config.cache.backend == "redis"
    assert config.cache.ttl == 3600
    
    assert isinstance(config.platforms, dict)
    assert "instagram" in config.platforms
    assert "tiktok" in config.platforms
    assert "youtube" in config.platforms

def test_load_from_yaml(config_file):
    """Test loading configuration from YAML file."""
    manager = ConfigurationManager(config_file)
    
    assert manager.config.database.host == "testdb.example.com"
    assert manager.config.database.port == 5432
    assert manager.config.database.database == "test_metrics"
    
    assert manager.config.cache.backend == "redis"
    assert manager.config.cache.host == "cache.example.com"
    assert manager.config.cache.ttl == 7200
    
    assert manager.config.platforms["instagram"].enabled is True
    assert manager.config.platforms["instagram"].api_key == "test_key"
    assert manager.config.platforms["instagram"].rate_limit == 200

def test_environment_variables():
    """Test environment variable integration."""
    os.environ["METRICS_DB_HOST"] = "envdb.

