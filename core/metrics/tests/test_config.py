import os
import tempfile
from pathlib import Path
import pytest
from dataclasses import dataclass
from typing import Optional, Dict, List

from core.metrics.config import (
    MetricsConfig,
    DatabaseConfig,
    CacheConfig,
    PlatformConfig,
    AlertConfig,
    RetentionConfig,
    ExportConfig,
    ConfigValidationError,
)

def test_cache_config_validation():
    """Test cache configuration validation rules"""
    # Test valid configuration
    valid_config = CacheConfig(
        enabled=True,
        ttl_seconds=3600,
        max_size_mb=1024,
        storage_path="/tmp/cache",
    )
    valid_config.validate()  # Should not raise

    # Test invalid TTL
    with pytest.raises(ConfigValidationError, match="TTL must be positive"):
        invalid_config = CacheConfig(
            enabled=True,
            ttl_seconds=-1,
            max_size_mb=1024,
            storage_path="/tmp/cache",
        )
        invalid_config.validate()

    # Test invalid max size
    with pytest.raises(ConfigValidationError, match="Max size must be positive"):
        invalid_config = CacheConfig(
            enabled=True,
            ttl_seconds=3600,
            max_size_mb=0,
            storage_path="/tmp/cache",
        )
        invalid_config.validate()

    # Test invalid storage path
    with pytest.raises(ConfigValidationError, match="Storage path must be absolute"):
        invalid_config = CacheConfig(
            enabled=True,
            ttl_seconds=3600,
            max_size_mb=1024,
            storage_path="relative/path",
        )
        invalid_config.validate()

def test_platform_config_validation():
    """Test platform configuration validation rules"""
    # Test valid configuration
    valid_config = PlatformConfig(
        enabled_platforms=["instagram", "tiktok", "youtube"],
        rate_limits={
            "instagram": 100,
            "tiktok": 200,
            "youtube": 150
        },
        timeout_seconds=30,
        retry_attempts=3,
    )
    valid_config.validate()  # Should not raise

    # Test invalid platform
    with pytest.raises(ConfigValidationError, match="Unknown platform"):
        invalid_config = PlatformConfig(
            enabled_platforms=["invalid_platform"],
            rate_limits={},
            timeout_seconds=30,
            retry_attempts=3,
        )
        invalid_config.validate()

    # Test missing rate limits
    with pytest.raises(ConfigValidationError, match="Rate limit required"):
        invalid_config = PlatformConfig(
            enabled_platforms=["instagram"],
            rate_limits={},
            timeout_seconds=30,
            retry_attempts=3,
        )
        invalid_config.validate()

    # Test invalid timeout
    with pytest.raises(ConfigValidationError, match="Timeout must be positive"):
        invalid_config = PlatformConfig(
            enabled_platforms=["instagram"],
            rate_limits={"instagram": 100},
            timeout_seconds=-1,
            retry_attempts=3,
        )
        invalid_config.validate()

def test_alert_config_validation():
    """Test alert configuration validation rules"""
    # Test valid configuration
    valid_config = AlertConfig(
        enabled=True,
        thresholds={
            "engagement_rate": 0.05,
            "viral_coefficient": 1.5,
            "growth_rate": 0.1
        },
        notification_channels=["email", "slack"],
        cooldown_minutes=15,
    )
    valid_config.validate()  # Should not raise

    # Test invalid threshold
    with pytest.raises(ConfigValidationError, match="Invalid threshold"):
        invalid_config = AlertConfig(
            enabled=True,
            thresholds={
                "engagement_rate": -0.1,  # Negative threshold
            },
            notification_channels=["email"],
            cooldown_minutes=15,
        )
        invalid_config.validate()

    # Test invalid notification channel
    with pytest.raises(ConfigValidationError, match="Unknown notification channel"):
        invalid_config = AlertConfig(
            enabled=True,
            thresholds={"engagement_rate": 0.05},
            notification_channels=["invalid_channel"],
            cooldown_minutes=15,
        )
        invalid_config.validate()

    # Test invalid cooldown
    with pytest.raises(ConfigValidationError, match="Cooldown must be positive"):
        invalid_config = AlertConfig(
            enabled=True,
            thresholds={"engagement_rate": 0.05},
            notification_channels=["email"],
            cooldown_minutes=0,
        )
        invalid_config.validate()

def test_retention_config_validation():
    """Test retention configuration validation rules"""
    # Test valid configuration
    valid_config = RetentionConfig(
        raw_data_days=30,
        aggregated_data_days=90,
        backup_enabled=True,
        backup_interval_hours=24,
        compression_enabled=True,
    )
    valid_config.validate()  # Should not raise

    # Test invalid retention period
    with pytest.raises(ConfigValidationError, match="Retention period must be positive"):
        invalid_config = RetentionConfig(
            raw_data_days=-1,
            aggregated_data_days=90,
            backup_enabled=True,
            backup_interval_hours=24,
            compression_enabled=True,
        )
        invalid_config.validate()

    # Test invalid backup interval
    with pytest.raises(ConfigValidationError, match="Backup interval must be positive"):
        invalid_config = RetentionConfig(
            raw_data_days=30,
            aggregated_data_days=90,
            backup_enabled=True,
            backup_interval_hours=0,
            compression_enabled=True,
        )
        invalid_config.validate()

def test_export_config_validation():
    """Test export configuration validation rules"""
    # Test valid configuration
    valid_config = ExportConfig(
        formats=["json", "csv"],
        export_path="/exports",
        schedule_enabled=True,
        schedule_cron="0 0 * * *",
        compression_enabled=True,
    )
    valid_config.validate()  # Should not raise

    # Test invalid format
    with pytest.raises(ConfigValidationError, match="Unknown export format"):
        invalid_config = ExportConfig(
            formats=["invalid_format"],
            export_path="/exports",
            schedule_enabled=True,
            schedule_cron="0 0 * * *",
            compression_enabled=True,
        )
        invalid_config.validate()

    # Test invalid export path
    with pytest.raises(ConfigValidationError, match="Export path must be absolute"):
        invalid_config = ExportConfig(
            formats=["json"],
            export_path="relative/path",
            schedule_enabled=True,
            schedule_cron="0 0 * * *",
            compression_enabled=True,
        )
        invalid_config.validate()

    # Test invalid cron expression
    with pytest.raises(ConfigValidationError, match="Invalid cron expression"):
        invalid_config = ExportConfig(
            formats=["json"],
            export_path="/exports",
            schedule_enabled=True,
            schedule_cron="invalid cron",
            compression_enabled=True,
        )
        invalid_config.validate()

def test_environment_override():
    """Test configuration overrides from environment variables"""
    # Set environment variables
    os.environ["METRICS_CACHE_ENABLED"] = "false"
    os.environ["METRICS_CACHE_TTL"] = "7200"
    os.environ["METRICS_PLATFORMS_ENABLED"] = "instagram,tiktok"
    os.environ["METRICS_ALERT_THRESHOLD_ENGAGEMENT_RATE"] = "0.1"

    config = MetricsConfig.load("config.yaml")
    
    # Verify environment overrides
    assert config.cache.enabled == False
    assert config.cache.ttl_seconds == 7200
    assert config.platform.enabled_platforms == ["instagram", "tiktok"]
    assert config.alert.thresholds["engagement_rate"] == 0.1

    # Clean up
    del os.environ["METRICS_CACHE_ENABLED"]
    del os.environ["METRICS_CACHE_TTL"]
    del os.environ["METRICS_PLATFORMS_ENABLED"]
    del os.environ["METRICS_ALERT_THRESHOLD_ENGAGEMENT_RATE"]

def test_config_inheritance():
    """Test configuration inheritance from base configuration"""
    base_config = {
        "cache": {
            "enabled": True,
            "ttl_seconds": 3600,
        },
        "platform": {
            "enabled_platforms": ["instagram"],
            "rate_limits": {"instagram": 100},
        }
    }

    override_config = {
        "cache": {
            "ttl_seconds": 7200,
        },
        "platform": {
            "enabled_platforms": ["instagram", "tiktok"],
        }
    }

    # Create temporary config files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as base_file, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as override_file:
        
        yaml.dump(base_config, base_file)
        yaml.dump(override_config, override_file)
        base_file.flush()
        override_file.flush()

        # Load configuration with inheritance
        config = MetricsConfig.load(
            override_file.name,
            base_config_path=base_file.name
        )

        # Verify inheritance
        assert config.cache.enabled == True  # Inherited from base
        assert config.cache.ttl_seconds == 7200  # Overridden
        assert set(config.platform.enabled_platforms) == {"instagram", "tiktok"}  # Merged
        assert config.platform.rate_limits["instagram"] == 100  # Inherited

def test_load_save():
    """Test complete configuration load and save functionality"""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        
        # Create test configuration
        config = MetricsConfig(
            cache=CacheConfig(
                enabled=True,
                ttl_seconds=3600,
                max_size_mb=1024,
                storage_path="/tmp/cache",
            ),
            platform=PlatformConfig(
                enabled_platforms=["instagram", "tiktok"],
                rate_limits={"instagram": 100, "tiktok": 200},
                timeout_seconds=30,
                retry_attempts=3,
            ),
            alert=AlertConfig(
                enabled=True,
                thresholds={"engagement_rate": 0.05},
                notification_channels=["email"],
                cooldown_minutes=15,
            ),
            retention=RetentionConfig(
                raw_data_days=30,
                aggregated_data_days=90,
                backup_enabled=True,
                backup_interval_hours=24,
                compression_enabled=True,
            ),
            export=ExportConfig(
                formats=["json", "csv"],
                export_path="/exports",
                schedule_enabled=True,
                schedule_cron="0 0 * * *",
                compression_enabled=True,
            ),
        )

        # Save configuration
        config.save(config_path)

        # Load configuration and verify
        loaded_config = MetricsConfig.load(config_path)

        # Verify all sections
        assert loaded_config.cache == config.cache
        assert loaded_config.platform == config.platform
        assert loaded_config.alert == config.alert
        assert loaded_config.retention == config.retention
        assert loaded_config.export == config.export

"""
Tests for the metrics configuration module.
"""

import os
from pathlib import Path
import pytest
import yaml

from core.metrics.config import (
    MetricsConfig,
    DatabaseConfig,
    CacheConfig,
    PlatformConfig,
    AlertConfig,
    RetentionConfig,
    ExportConfig,
    ConfigValidationError,
    ConfigLoadError,
)

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config_data = {
        "database": {
            "host": "test-host",
            "port": 5432,
            "name": "test-db",
            "user": "test-user",
            "password": "test-pass",
        },
        "cache": {
            "enabled": True,
            "backend": "redis",
            "host": "cache-host",
            "port": 6379,
        },
        "platforms": {
            "instagram": {
                "api_key": "test-key",
                "api_secret": "test-secret",
                "rate_limit": 100,
            },
            "tiktok": {
                "api_key": "tiktok-key",
                "api_secret": "tiktok-secret",
                "rate_limit": 200,
            },
        },
        "alerts": {
            "enabled": True,
            "viral_coefficient_threshold": 2.0,
            "notification_channels": ["email", "slack"],
        },
        "retention": {
            "raw_data_days": 60,
            "aggregated_data_months": 24,
        },
        "export": {
            "formats": ["json", "csv"],
            "compression": True,
            "export_path": "/exports",
        },
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return config_file

def test_database_config_validation():
    """Test database configuration validation."""
    # Valid configuration
    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        name="metrics",
        user="metrics_user",
    )
    db_config.validate()  # Should not raise
    
    # Invalid port
    with pytest.raises(ConfigValidationError):
        db_config = DatabaseConfig(port=0)
        db_config.validate()
    
    # Empty host
    with pytest.raises(ConfigValidationError):
        db_config = DatabaseConfig(host="")
        db_config.validate()

def test_cache_config_validation():
    """Test cache configuration validation."""
    # Valid configuration
    cache_config = CacheConfig(
        enabled=True,
        backend="redis",
        host="localhost",
    )
    cache_config.validate()  # Should not raise
    
    # Invalid backend
    with pytest.raises(ConfigValidationError):
        cache_config = CacheConfig(backend="invalid")
        cache_config.validate()
    
    # Invalid port
    with pytest.raises(ConfigValidationError):
        

import os
import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.metrics.config import (
    MetricsConfig,
    DatabaseConfig,
    CacheConfig,
    PlatformConfig,
    AlertConfig,
    RetentionConfig,
    PerformanceConfig,
    ExportConfig,
    ConfigValidationError,
)

class TestMetricsConfig:
    @pytest.fixture
    def base_config(self):
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "metrics_db",
                "user": "metrics_user",
                "password": "default_password",
            },
            "cache": {
                "enabled": True,
                "ttl_seconds": 300,
                "max_size_mb": 100,
            },
            "platforms": {
                "instagram": {"api_key": "default_ig_key", "rate_limit": 100},
                "tiktok": {"api_key": "default_tt_key", "rate_limit": 200},
                "youtube": {"api_key": "default_yt_key", "rate_limit": 150},
            },
            "alerts": {
                "engagement_threshold": 0.1,
                "viral_coefficient": 1.5,
                "notification_email": "alerts@example.com",
            },
        }

    @pytest.fixture
    def config_file(self, base_config):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(base_config, f)
            return Path(f.name)

    def test_environment_variables_override(self, base_config, config_file):
        with patch.dict(os.environ, {
            'METRICS_DB_HOST': 'env_host',
            'METRICS_DB_PASSWORD': 'env_password',
            'METRICS_CACHE_TTL': '600',
            'METRICS_INSTAGRAM_API_KEY': 'env_ig_key',
        }):
            config = MetricsConfig.from_file(config_file)
            assert config.database.host == 'env_host'
            assert config.database.password == 'env_password'
            assert config.cache.ttl_seconds == 600
            assert config.platforms['instagram'].api_key == 'env_ig_key'
            # Verify non-overridden values remain unchanged
            assert config.database.port == base_config['database']['port']
            assert config.platforms['tiktok'].api_key == base_config['platforms']['tiktok']['api_key']

    def test_config_validation(self, base_config, config_file):
        # Test invalid database configuration
        invalid_db = base_config.copy()
        invalid_db['database']['port'] = "not_an_integer"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_db, f)
            with pytest.raises(ConfigValidationError, match="Database port must be an integer"):
                MetricsConfig.from_file(Path(f.name))

        # Test invalid cache configuration
        invalid_cache = base_config.copy()
        invalid_cache['cache']['ttl_seconds'] = -1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_cache, f)
            with pytest.raises(ConfigValidationError, match="Cache TTL must be positive"):
                MetricsConfig.from_file(Path(f.name))

    def test_config_reload(self, base_config, config_file):
        config = MetricsConfig.from_file(config_file)
        
        # Modify configuration file
        modified_config = base_config.copy()
        modified_config['database']['host'] = 'modified_host'
        modified_config['cache']['ttl_seconds'] = 900
        
        with open(config_file, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Test reload
        config.reload()
        assert config.database.host == 'modified_host'
        assert config.cache.ttl_seconds == 900

    def test_all_configuration_sections(self, base_config, config_file):
        config = MetricsConfig.from_file(config_file)
        
        # Test database configuration
        assert isinstance(config.database, DatabaseConfig)
        assert config.database.host == base_config['database']['host']
        assert config.database.port == base_config['database']['port']
        
        # Test cache configuration
        assert isinstance(config.cache, CacheConfig)
        assert config.cache.enabled == base_config['cache']['enabled']
        assert config.cache.ttl_seconds == base_config['cache']['ttl_seconds']
        
        # Test platform configurations
        for platform in ['instagram', 'tiktok', 'youtube']:
            assert isinstance(config.platforms[platform], PlatformConfig)
            assert config.platforms[platform].api_key == base_config['platforms'][platform]['api_key']
            assert config.platforms[platform].rate_limit == base_config['platforms'][platform]['rate_limit']

        # Test alert configuration
        assert isinstance(config.alerts, AlertConfig)
        assert config.alerts.engagement_threshold == base_config['alerts']['engagement_threshold']
        assert config.alerts.viral_coefficient == base_config['alerts']['viral_coefficient']

    def test_error_handling(self, base_config):
        # Test missing required fields
        incomplete_config = {"database": {"host": "localhost"}}  # Missing required fields
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(incomplete_config, f)
            with pytest.raises(ConfigValidationError, match="Missing required database configuration"):
                MetricsConfig.from_file(Path(f.name))

        # Test invalid file path
        with pytest.raises(FileNotFoundError):
            MetricsConfig.from_file(Path("nonexistent_config.yaml"))

        # Test invalid YAML syntax
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: {")
            with pytest.raises(yaml.YAMLError):
                MetricsConfig.from_file(Path(f.name))

    def test_config_inheritance(self, base_config):
        parent_config = base_config.copy()
        child_config = {
            "database": {"host": "child_host"},
            "platforms": {
                "instagram": {"rate_limit": 150},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(parent_config, f)
            parent_path = Path(f.name)
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            child_config['inherit_from'] = str(parent_path)
            yaml.dump(child_config, f)
            config = MetricsConfig.from_file(Path(f.name))
            
            # Verify inheritance
            assert config.database.host == "child_host"  # Overridden
            assert config.database.port == parent_config['database']['port']  # Inherited
            assert config.platforms['instagram'].rate_limit == 150  # Overridden
            assert config.platforms['instagram'].api_key == parent_config['platforms']['instagram']['api_key']  # Inherited

    def test_config_export(self, base_config, config_file):
        config = MetricsConfig.from_file(config_file)
        
        # Test YAML export
        export_path = Path(tempfile.gettempdir()) / "exported_config.yaml"
        config.export(export_path, format="yaml")
        
        # Verify exported configuration
        with open(export_path) as f:
            exported_config = yaml.safe_load(f)
            assert exported_config['database']['host'] == base_config['database']['host']
            assert exported_config['cache']['ttl_seconds'] == base_config['cache']['ttl_seconds']
        
        # Test JSON export
        export_path = Path(tempfile.gettempdir()) / "exported_config.json"
        config.export(export_path, format="json")
        
        # Verify JSON export
        with open(export_path) as f:
            import json
            exported_config = json.load(f)
            assert exported_config['database']['host'] == base_config['database']['host']
            assert exported_config['cache']['ttl_seconds'] == base_config['cache']['ttl_seconds']

