import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch

from ..config import MetricsConfig, DatabaseConfig, CacheConfig, PlatformConfig, AlertConfig
from ..persistence import MetricsPersistence
from ..metrics_aggregator import MetricsAggregator, MetricsSnapshot

class TestMetricsIntegration:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.yaml"
        self.db_path = Path(self.test_dir) / "test.db"
        
        # Setup basic configuration
        self.base_config = {
            "database": {
                "path": str(self.db_path),
                "backup_path": str(Path(self.test_dir) / "backups"),
                "retention_days": 30
            },
            "cache": {
                "enabled": True,
                "max_size_mb": 100,
                "ttl_seconds": 3600
            },
            "platforms": {
                "instagram": {"enabled": True, "api_key": "test_key"},
                "tiktok": {"enabled": True, "api_key": "test_key"},
                "youtube": {"enabled": True, "api_key": "test_key"}
            },
            "alerts": {
                "engagement_rate_threshold": 0.05,
                "viral_coefficient_threshold": 1.5,
                "notification_email": "test@example.com"
            }
        }
        
        # Write initial configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.base_config, f)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir)
    
    def test_config_persistence_flow(self):
        """Test complete configuration flow with persistence"""
        # Initialize config from file
        config = MetricsConfig.from_file(self.config_path)
        assert config.database.path == str(self.db_path)
        
        # Modify and save configuration
        config.cache.max_size_mb = 200
        config.save(self.config_path)
        
        # Reload and verify changes persisted
        new_config = MetricsConfig.from_file(self.config_path)
        assert new_config.cache.max_size_mb == 200
    
    def test_env_override(self):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {
            'METRICS_DB_PATH': '/custom/path/db.sqlite',
            'METRICS_CACHE_ENABLED': 'false',
            'METRICS_INSTAGRAM_API_KEY': 'env_key'
        }):
            config = MetricsConfig.from_file(self.config_path)
            assert config.database.path == '/custom/path/db.sqlite'
            assert not config.cache.enabled
            assert config.platforms['instagram'].api_key == 'env_key'
    
    def test_config_inheritance(self):
        """Test configuration inheritance and validation"""
        # Create base config
        base_config = {
            "database": {"retention_days": 30},
            "cache": {"enabled": True}
        }
        
        # Create child config that inherits and overrides
        child_config = {
            "inherit_from": str(self.config_path),
            "database": {"retention_days": 60}
        }
        
        child_path = Path(self.test_dir) / "child_config.yaml"
        with open(child_path, 'w') as f:
            json.dump(child_config, f)
        
        config = MetricsConfig.from_file(child_path)
        assert config.database.retention_days == 60
        assert config.cache.enabled  # Inherited from base
    
    def test_metrics_storage_retrieval(self):
        """Test metrics storage and retrieval through configuration"""
        config = MetricsConfig.from_file(self.config_path)
        persistence = MetricsPersistence(config)
        
        # Store test metrics
        metrics = MetricsSnapshot(
            timestamp=datetime.now(),
            platform="instagram",
            engagement_rate=0.1,
            viral_coefficient=1.2,
            reach=1000,
            interactions=100
        )
        persistence.store_metrics(metrics)
        
        # Retrieve and verify
        stored = persistence.get_metrics_by_platform("instagram", 
                                                   from_time=datetime.now() - timedelta(hours=1))
        assert len(stored) == 1
        assert stored[0].engagement_rate == 0.1
    
    def test_export_functionality(self):
        """Test export functionality with different formats"""
        config = MetricsConfig.from_file(self.config_path)
        persistence = MetricsPersistence(config)
        
        # Store some test data
        for i in range(3):
            metrics = MetricsSnapshot(
                timestamp=datetime.now() - timedelta(hours=i),
                platform="tiktok",
                engagement_rate=0.1 * i,
                viral_coefficient=1.0 + i,
                reach=1000 * i,
                interactions=100 * i
            )
            persistence.store_metrics(metrics)
        
        # Test JSON export
        json_path = Path(self.test_dir) / "export.json"
        persistence.export_metrics(json_path, format="json")
        with open(json_path) as f:
            exported = json.load(f)
            assert len(exported) == 3
        
        # Test CSV export
        csv_path = Path(self.test_dir) / "export.csv"
        persistence.export_metrics(csv_path, format="csv")
        with open(csv_path) as f:
            lines = f.readlines()
            assert len(lines) == 4  # Header + 3 data rows
    
    def test_alert_monitoring(self):
        """Test alert threshold monitoring"""
        config = MetricsConfig.from_file(self.config_path)
        aggregator = MetricsAggregator(config)
        
        # Test metrics that should trigger alert
        metrics = MetricsSnapshot(
            timestamp=datetime.now(),
            platform="youtube",
            engagement_rate=0.06,  # Above threshold
            viral_coefficient=2.0,  # Above threshold
            reach=5000,
            interactions=500
        )
        
        with patch('core.metrics.metrics_aggregator.send_alert') as mock_send_alert:
            aggregator.process_metrics(metrics)
            assert mock_send_alert.called
            
            # Verify alert content
            alert_args = mock_send_alert.call_args[0]
            assert "engagement_rate" in alert_args[0]
            assert "viral_coefficient" in alert_args[0]
    
    def test_backup_retention(self):
        """Test backup and retention policies"""
        config = MetricsConfig.from_file(self.config_path)
        persistence = MetricsPersistence(config)
        
        # Create some test data
        old_date = datetime.now() - timedelta(days=40)
        recent_date = datetime.now() - timedelta(days=5)
        
        with patch('core.metrics.persistence.datetime') as mock_datetime:
            # Insert old data
            mock_datetime.now.return_value = old_date
            persistence.store_metrics(MetricsSnapshot(
                timestamp=old_date,
                platform="instagram",
                engagement_rate=0.1,
                viral_coefficient=1.2,
                reach=1000,
                interactions=100
            ))
            
            # Insert recent data
            mock_datetime.now.return_value = recent_date
            persistence.store_metrics(MetricsSnapshot(
                timestamp=recent_date,
                platform="instagram",
                engagement_rate=0.2,
                viral_coefficient=1.4,
                reach=2000,
                interactions=200
            ))
        
        # Run retention cleanup
        persistence.cleanup_old_data()
        
        # Verify only recent data remains
        stored = persistence.get_all_metrics()
        assert len(stored) == 1
        assert stored[0].reach == 2000  # Only recent data should remain

