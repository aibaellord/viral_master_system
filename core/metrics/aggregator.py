from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Union
import asyncio

from .types import (
    MetricType,
    AlertConfig,
    InsightType,
    TrendData,
    AggregateStats,
)
from .config import MetricsConfig
from .persistence import MetricsPersistence

class MetricsInsight:
    def __init__(self, type: InsightType, description: str, confidence: float, data: Dict[str, Any]):
        self.type = type
        self.description = description
        self.confidence = confidence
        self.data = data

class MetricsAggregator:
    def __init__(self, config: MetricsConfig, persistence: MetricsPersistence, platform_clients: List[Any]):
        self.config = config
        self.persistence = persistence
        self.platform_clients = {client.platform_name: client for client in platform_clients}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on config settings"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )

    def _get_cache_key(self, metric: MetricType) -> str:
        """Generate a unique cache key for a metric"""
        return f"{metric.platform}:{metric.type}:{metric.id}"

    def _update_cache(self, metrics: List[MetricType]):
        """Update the cache with new metrics"""
        current_time = datetime.utcnow()
        for metric in metrics:
            cache_key = self._get_cache_key(metric)
            self.cache[cache_key] = {
                'value': metric.value,
                'timestamp': current_time,
                'ttl': current_time + timedelta(seconds=self.config.cache.ttl)
            }

    def _clean_cache(self):
        """Remove expired items from cache"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, data in self.cache.items()
            if data['ttl'] < current_time
        ]
        for key in expired_keys:
            del self.cache[key]

    async def _check_alerts(self, metrics: List[MetricType]):
        """Check metrics against alert configurations and trigger notifications"""
        for alert_config in self.config.alerts:
            for metric in metrics:
                if metric.type == alert_config.metric_type:
                    triggered = False
                    if alert_config.comparison == "greater_than":
                        triggered = metric.value > alert_config.threshold
                    elif alert_config.comparison == "less_than":
                        triggered = metric.value < alert_config.threshold

                    if triggered:
                        for notifier in alert_config.notifiers:
                            try:
                                await notifier.send_alert(metric, alert_config)
                            except Exception as e:
                                self.logger.error(f"Failed to send alert: {e}")

    async def process_incoming_metrics(self, metrics: List[MetricType], platform: str):
        """Process incoming metrics, store them, and update cache"""
        try:
            # Store metrics in persistence layer
            await self.persistence.store_metrics(metrics)
            
            # Update cache
            self._update_cache(metrics)
            
            # Clean expired cache entries
            self._clean_cache()
            
            # Check for alerts
            await self._check_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"Error processing metrics: {e}")
            raise

    async def calculate_trends(
        self,
        metric_type: str,
        time_window: timedelta,
        granularity: str
    ) -> TrendData:
        """Calculate trends for specified metric type and time window"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            # Get metrics from persistence layer
            metrics = await self.persistence.get_metrics_range(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                granularity=granularity
            )
            
            # Calculate trend statistics
            values = [metric.value for metric in metrics]
            if not values:
                return TrendData(
                    metric_type=metric_type,
                    start_time=start_time,
                    end_time=end_time,
                    trend_direction="none",
                    change_rate=0.0,
                    confidence=0.0
                )
            
            avg_value = sum(values) / len(values)
            trend_direction = "up" if values[-1] > avg_value else "down"
            change_rate = ((values[-1] - values[0]) / values[0]) if values[0] != 0 else 0
            
            return TrendData(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                trend_direction=trend_direction,
                change_rate=change_rate,
                confidence=0.95
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating trends: {e}")
            raise

    async def generate_insights(
        self,
        metric_types: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None
    ) -> List[MetricsInsight]:
        """Generate insights from collected metrics"""
        try:
            insights = []
            all_metrics = await self.persistence.get_recent_metrics(
                metric_types=metric_types,
                platforms=platforms,
                limit=1000
            )
            
            # Group metrics by type and platform
            grouped_metrics = {}
            for metric in all_metrics:
                key = (metric.type, metric.platform)
                if key not in grouped_metrics:
                    grouped_metrics[key] = []
                grouped_metrics[key].append(metric)
            
            # Generate insights for each group
            for (metric_type, platform), metrics in grouped_metrics.items():
                if len(metrics) < 2:
                    continue
                    
                values = [m.value for m in metrics]
                avg = sum(values) / len(values)
                std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
                
                # Check for anomalies
                if abs(values[-1] - avg) > 2 * std_dev:
                    insights.append(MetricsInsight(
                        type=InsightType.ANOMALY,
                        description=f"Anomaly detected in {metric_type} for {platform}",
                        confidence=0.95,
                        data={
                            "metric_type": metric_type,
                            "platform": platform,
                            "value": values[-1],
                            "average": avg,
                            "std_dev": std_dev
                        }
                    ))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            raise

    async def get_aggregate_stats(
        self,
        metric_type: str,
        platform: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> AggregateStats:
        """Get aggregate statistics for specified metrics"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - (time_window or timedelta(days=7))
            
            metrics = await self.persistence.get_metrics_range(
                metric_type=metric_type,
                platform=platform,
                start_time=start_time,
                end_time=end_time
            )
            
            if not metrics:
                return AggregateStats(
                    metric_type=metric_type,
                    platform=platform,
                    count=0,
                    min_value=0,
                    max_value=0,
                    avg_value=0,
                    std_dev=0
                )
            
            values = [m.value for m in metrics]
            avg = sum(values) / len(values)
            std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
            
            return AggregateStats(
                metric_type=metric_type,
                platform=platform,
                count=len(metrics),
                min_value=min(values),
                max_value=max(values),
                avg_value=avg,
                std_dev=std_dev
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating aggregate stats: {e}")
            raise

