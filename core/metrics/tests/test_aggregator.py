import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from ..aggregator import MetricsAggregator, MetricsInsight
from ..types import (
    MetricType,
    AlertConfig,
    InsightType,
    TrendData,
    AggregateStats,
)
from ..config import MetricsConfig
from ..persistence import MetricsPersistence

@pytest.fixture
def mock_config():
    config = Mock(spec=MetricsConfig)
    config.cache.ttl = 3600
    config.retention.days = 30
    config.logging.level = "INFO"
    config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    config.alerts = []
    return config

@pytest.fixture
def mock_persistence():
    persistence = AsyncMock(spec=MetricsPersistence)
    return persistence

@pytest.fixture
def mock_platform_clients():
    client1 = AsyncMock()
    client1.platform_name = "instagram"
    client2 = AsyncMock()
    client2.platform_name = "tiktok"
    return [client1, client2]

@pytest.fixture
async def aggregator(mock_config, mock_persistence, mock_platform_clients):
    return MetricsAggregator(mock_config, mock_persistence, mock_platform_clients)

@pytest.mark.asyncio
async def test_process_incoming_metrics(aggregator, mock_persistence):
    # Prepare test data
    test_metrics = [
        Mock(spec=MetricType),
        Mock(spec=MetricType),
    ]
    
    # Configure mocks
    for metric in test_metrics:
        metric.timestamp = datetime.utcnow()
        metric.value = 100
        metric.platform = "instagram"
        metric.type = "engagement"
        metric.id = "test_id"
        metric.min_value = 0
        metric.max_value = 1000

    # Process metrics
    await aggregator.process_incoming_metrics(test_metrics, "instagram")
    
    # Verify persistence layer was called
    mock_persistence.store_metrics.assert_called_once()
    assert len(aggregator.cache) == len(test_metrics)

@pytest.mark.asyncio
async def test_alert_triggering(aggregator):
    # Configure alert
    alert_config = AlertConfig(
        metric_type="engagement",
        comparison="greater_than",
        threshold=90,
        notifiers=[AsyncMock()]
    )
    aggregator.config.alerts = [alert_config]
    
    # Create test metric that should trigger alert
    test_metric = Mock(spec=MetricType)
    test_metric.type = "engagement"
    test_metric.value = 100
    test_metric.timestamp = datetime.utcnow()
    test_metric.platform = "instagram"
    
    # Process metric
    await aggregator._check_alerts([test_metric])
    
    # Verify alert was triggered
    alert_config.notifiers[0].send_alert.assert_called_once()

@pytest.mark.asyncio
async def test_trend_calculation(aggregator, mock_persistence):
    # Prepare test data
    start_time = datetime.utcnow() - timedelta(days=1)
    end_time = datetime.utcnow()
    test_metrics = [
        Mock(spec=MetricType),
        Mock(spec=MetricType),
    ]
    
    # Configure mock persistence
    mock_persistence.get_metrics_range.return_value = test_metrics
    
    # Calculate trends
    trend_data = await aggregator.calculate_trends(
        "engagement",
        timedelta(days=1),
        "1h"
    )
    
    # Verify persistence was called correctly
    mock_persistence.get_metrics_range.assert_called_once()
    assert isinstance(trend_data, TrendData)

@pytest.mark.asyncio
async def test_cache_management(aggregator):
    # Add items to cache
    test_metric = Mock(spec=MetricType)
    test_metric.platform = "instagram"
    test_metric.type = "engagement"
    test_metric.id = "test_id"
    test_metric.value = 100
    test_metric.timestamp = datetime.utcnow()
    
    # Update cache
    aggregator._update_cache([test_metric])
    
    # Verify cache contains item
    cache_key = aggregator._get_cache_key(test_metric)
    assert cache_key in aggregator.cache
    assert aggregator.cache[cache_key]['value'] == 100
    
    #

