from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from .config import MetricsConfig
from .persistence import MetricsPersistence
from ..platforms.base_platform_client import BasePlatformClient
from ..platforms.instagram_client import InstagramClient
from ..platforms.tiktok_client import TikTokClient
from ..platforms.youtube_client import YouTubeClient

@dataclass
class MetricsSnapshot:
    """Snapshot of metrics across all platforms at a specific point in time."""
    timestamp: datetime
    snapshot_id: UUID
    platform_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engagement_rates: Dict[str, float] = field(default_factory=dict)
    viral_coefficients: Dict[str, float] = field(default_factory=dict)
    growth_rates: Dict[str, float] = field(default_factory=dict)
    audience_retention: Dict[str, float] = field(default_factory=dict)
    content_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cross_platform_impact: float = 0.0
    alert_triggers: List[str] = field(default_factory=list)

class MetricsAggregator:
    """Handles collection, processing, and monitoring of metrics across platforms."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.persistence = MetricsPersistence(config.database)
        self._initialize_platform_clients()
        self._load_alert_thresholds()
        
    def _initialize_platform_clients(self):
        """Initialize platform-specific API clients."""
        self.platform_clients: Dict[str, BasePlatformClient] = {}
        if self.config.platforms.instagram_enabled:
            self.platform_clients['instagram'] = InstagramClient(
                self.config.platforms.instagram_credentials
            )
        if self.config.platforms.tiktok_enabled:
            self.platform_clients['tiktok'] = TikTokClient(
                self.config.platforms.tiktok_credentials
            )
        if self.config.platforms.youtube_enabled:
            self.platform_clients['youtube'] = YouTubeClient(
                self.config.platforms.youtube_credentials
            )

    def _load_alert_thresholds(self):
        """Load alert thresholds from configuration."""
        self.alert_thresholds = self.config.alerts.thresholds
        
    async def collect_metrics(self) -> MetricsSnapshot:
        """Collect metrics from all enabled platforms."""
        snapshot = MetricsSnapshot(
            timestamp=datetime.utcnow(),
            snapshot_id=UUID.uuid4()
        )
        
        for platform, client in self.platform_clients.items():
            try:
                metrics = await client.get_metrics()
                snapshot.platform_metrics[platform] = metrics
                snapshot.engagement_rates[platform] = await self._calculate_engagement_rate(
                    platform, metrics
                )
                snapshot.viral_coefficients[platform] = await self._calculate_viral_coefficient(
                    platform, metrics
                )
                snapshot.growth_rates[platform] = await self._calculate_growth_rate(
                    platform, metrics
                )
                snapshot.audience_retention[platform] = await self._calculate_audience_retention(
                    platform, metrics
                )
                snapshot.content_performance[platform] = await self._analyze_content_performance(
                    platform, metrics
                )
            except Exception as e:
                self._handle_collection_error(platform, e)
                
        snapshot.cross_platform_impact = self._calculate_cross_platform_impact(snapshot)
        snapshot.alert_triggers = self._check_alert_conditions(snapshot)
        
        await self.persistence.store_snapshot(snapshot)
        return snapshot
    
    async def _calculate_engagement_rate(
        self, platform: str, metrics: Dict[str, float]
    ) -> float:
        """Calculate platform-specific engagement rate."""
        if platform == 'instagram':
            return (metrics['likes'] + metrics['comments']) / metrics['followers'] * 100
        elif platform == 'tiktok':
            return (metrics['likes'] + metrics['comments'] + metrics['shares']) / metrics['views'] * 100
        elif platform == 'youtube':
            return (metrics['likes'] + metrics['comments']) / metrics['views'] * 100
        return 0.0

    async def _calculate_viral_coefficient(
        self, platform: str, metrics: Dict[str, float]
    ) -> float:
        """Calculate viral coefficient for content spread."""
        if platform == 'tiktok':
            return metrics['shares'] * metrics['conversion_rate']
        elif platform == 'youtube':
            return metrics['shares'] * metrics['subscriber_conversion_rate']
        elif platform == 'instagram':
            return metrics['saves'] * metrics['follower_conversion_rate']
        return 0.0

    async def _calculate_growth_rate(
        self, platform: str, metrics: Dict[str, float]
    ) -> float:
        """Calculate platform-specific growth rate."""
        previous_metrics = await self.persistence.get_previous_metrics(platform)
        if not previous_metrics:
            return 0.0
        
        current_followers = metrics['followers']
        previous_followers = previous_metrics['followers']
        time_diff = metrics['timestamp'] - previous_metrics['timestamp']
        
        return ((current_followers - previous_followers) / previous_followers) * (
            86400 / time_diff.total_seconds()
        )  # Normalize to daily rate

    async def _calculate_audience_retention(
        self, platform: str, metrics: Dict[str, float]
    ) -> float:
        """Calculate audience retention rate."""
        if platform == 'youtube':
            return metrics['average_view_duration'] / metrics['video_length'] * 100
        elif platform == 'tiktok':
            return metrics['video_completion_rate'] * 100
        elif platform == 'instagram':
            return metrics['story_completion_rate'] * 100
        return 0.0

    async def _analyze_content_performance(
        self, platform: str, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze content performance metrics."""
        return {
            'reach_rate': metrics['reach'] / metrics['followers'] * 100,
            'engagement_quality': self._calculate_engagement_quality(metrics),
            'virality_score': self._calculate_virality_score(platform, metrics),
            'audience_growth': self._calculate_audience_growth_rate(metrics),
            'content_consistency': self._calculate_content_consistency(metrics)
        }

    def _calculate_cross_platform_impact(self, snapshot: MetricsSnapshot) -> float:
        """Calculate overall cross-platform impact score."""
        weights = self.config.analytics.platform_weights
        impact_score = 0.0
        total_weight = 0.0
        
        for platform in snapshot.platform_metrics:
            weight = weights.get(platform, 1.0)
            impact_score += (
                snapshot.viral_coefficients[platform] * 0.4 +
                snapshot.engagement_rates[platform] * 0.3 +
                snapshot.growth_rates[platform] * 0.3
            ) * weight
            total_weight += weight
            
        return impact_score / total_weight if total_weight > 0 else 0.0

    def _check_alert_conditions(self, snapshot: MetricsSnapshot) -> List[str]:
        """Check for alert conditions in metrics."""
        alerts = []
        
        for platform, metrics in snapshot.platform_metrics.items():
            for metric, threshold in self.alert_thresholds.items():
                if metric in metrics and metrics[metric] < threshold:
                    alerts.append(
                        f"{platform}_{metric}_below_threshold_{threshold}"
                    )
                    
        if snapshot.cross_platform_impact < self.alert_thresholds.get(
            'cross_platform_impact', 0
        ):
            alerts.append("low_cross_platform_impact")
            
        return alerts

    def _handle_collection_error(self, platform: str, error: Exception):
        """Handle errors during metrics collection."""
        error_msg = f"Error collecting metrics for {platform}: {str(error)}"
        if self.config.logging.enabled:
            self.persistence.log_error(error_msg)
        if self.config.alerts.error_notifications:
            self._send_error_notification(platform, error_msg)
            
    async def get_historical_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        platforms: Optional[List[str]] = None
    ) -> List[MetricsSnapshot]:
        """Retrieve historical metrics within the specified time range."""
        return await self.persistence.get_metrics_range(start_time, end_time, platforms)

    async def generate_analytics_report(
        self,
        snapshot: MetricsSnapshot,
        report_type: str = 'full'
    ) -> Dict[str, any]:
        """Generate analytics report from metrics snapshot."""
        report = {
            'timestamp': snapshot.timestamp,
            'snapshot_id': snapshot.snapshot_id,
            'cross_platform_impact': snapshot.cross_platform_impact,
            'platforms': {}
        }
        
        for platform in snapshot.platform_metrics:
            report['platforms'][platform] = {
                'metrics': snapshot.platform_metrics[platform],
                'engagement_rate': snapshot.engagement_rates[platform],
                'viral_coefficient': snapshot.viral_coefficients[platform],
                'growth_rate': snapshot.growth_rates[platform],
                'audience_retention': snapshot.audience_retention[platform],
                'content_performance': snapshot.content_performance[platform]
            }
            
        if report_type == 'full':
            report['trend_analysis'] = await self._generate_trend_analysis(snapshot)
            report['recommendations'] = await self._generate_recommendations(snapshot)
            
        return report

    async def _generate_trend_analysis(
        self, snapshot: MetricsSnapshot
    ) -> Dict[str, any]:
        """Generate trend analysis from historical data."""
        historical_data = await self.get_historical_metrics(
            start_time=snapshot.timestamp - self.config.analytics.trend_window,
            end_time=snapshot.timestamp
        )
        
        return {
            'growth_trends': self._analyze_growth_trends(historical_data),
            'engagement_patterns': self._analyze_engagement_patterns(historical_data),
            'viral_patterns': self._analyze_viral_patterns(historical_data),
            'audience_behavior': self._analyze_audience_behavior(historical_data)
        }

    async def _generate_recommendations(
        self, snapshot: MetricsSnapshot
    ) -> List[str]:
        """Generate recommendations based on metrics analysis."""
        recommendations = []
        
        for platform, metrics in snapshot.platform_metrics.items():
            if snapshot.engagement_rates[platform] < self.config.analytics.engagement_threshold:
                recommendations.append(
                    f"Increase engagement on {platform} through more interactive content"
                )
            if snapshot.viral_coefficients[platform] < self.config.analytics.virality_threshold:
                recommendations.append(
                    f"Optimize content virality on {platform} through better call-to-actions"
                )
                
        return recommendations

    def cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        retention_days = self.config.retention.days
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        self.persistence.cleanup_old_metrics(cutoff_date)

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import asyncio
import json
import numpy as np
from pathlib import Path
import redis
from redis.client import Redis

from core.platforms.base_platform_client import BasePlatformClient
from core.platforms.instagram_client import InstagramClient
from core.platforms.tiktok_client import TikTokClient
from core.platforms.youtube_client import YouTubeClient

@dataclass
class MetricsSnapshot:
    timestamp: datetime
    engagement_rate: float
    viral_coefficient: float
    reach: int
    impressions: int
    shares: int
    likes: int
    comments: int
    saves: int
    watch_time: float
    retention_rate: float
    follower_growth: int 
    sentiment_score: float
    platform_metrics: Dict[str, Dict[str, float]]

class MetricsCache:
    def __init__(self, redis_url: str):
        self.redis: Redis = redis.from_url(redis_url)
        self.ttl = timedelta(hours=24)
    
    def get(self, key: str) -> Optional[MetricsSnapshot]:
        data = self.redis.get(key)
        if data:
            return MetricsSnapshot(**json.loads(data))
        return None
    
    def set(self, key: str, snapshot: MetricsSnapshot):
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(snapshot.__dict__)
        )

class AlertThresholds:
    def __init__(self):
        self.engagement_rate_min = 0.02
        self.viral_coefficient_min = 1.0
        self.sentiment_score_min = 0.6
        self.retention_rate_min = 0.4

class MetricsAggregator:
    def __init__(
        self,
        instagram_client: InstagramClient,
        tiktok_client: TikTokClient,
        youtube_client: YouTubeClient,
        redis_url: str,
        storage_path: Path
    ):
        self.platform_clients: Dict[str, BasePlatformClient] = {
            "instagram": instagram_client,
            "tiktok": tiktok_client,
            "youtube": youtube_client
        }
        self.cache = MetricsCache(redis_url)
        self.storage_path = storage_path
        self.alert_thresholds = AlertThresholds()
        self.monitoring = False
        self._setup_storage()

    def _setup_storage(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def collect_platform_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for platform, client in self.platform_clients.items():
            platform_metrics = await client.get_metrics()
            normalized_metrics = self._normalize_metrics(platform, platform_metrics)
            metrics[platform] = normalized_metrics
        return metrics

    def _normalize_metrics(
        self,
        platform: str,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Normalize platform-specific metrics to common scale."""
        if platform == "instagram":
            return {
                "engagement": metrics["engagement_rate"] * 100,
                "reach": metrics["reach_rate"] * 100,
                "virality": metrics["save_rate"] * 1.5
            }
        elif platform == "tiktok":
            return {
                "engagement": metrics["interaction_rate"] * 120,
                "reach": metrics["view_rate"] * 80,
                "virality": metrics["share_rate"] * 2.0
            }
        else:  # youtube
            return {
                "engagement": metrics["watch_percentage"] * 90,
                "reach": metrics["impression_rate"] * 70,
                "virality": metrics["subscriber_conversion"] * 1.8
            }

    async def create_snapshot(self) -> MetricsSnapshot:
        platform_metrics = await self.collect_platform_metrics()
        
        # Calculate aggregate metrics
        total_engagement = sum(m["engagement"] for m in platform_metrics.values())
        total_reach = sum(m["reach"] for m in platform_metrics.values())
        avg_virality = np.mean([m["virality"] for m in platform_metrics.values()])
        
        raw_metrics = await asyncio.gather(*[
            client.get_raw_metrics()
            for client in self.platform_clients.values()
        ])
        
        total_metrics = {
            "likes": sum(m["likes"] for m in raw_metrics),
            "comments": sum(m["comments"] for m in raw_metrics),
            "shares": sum(m["shares"] for m in raw_metrics),
            "saves": sum(m["saves"] for m in raw_metrics),
            "watch_time": sum(m["watch_time"] for m in raw_metrics),
            "followers_delta": sum(m["followers_change"] for m in raw_metrics)
        }
        
        return MetricsSnapshot(
            timestamp=datetime.now(),
            engagement_rate=total_engagement / len(platform_metrics),
            viral_coefficient=avg_virality,
            reach=total_reach,
            impressions=sum(m["impressions"] for m in raw_metrics),
            shares=total_metrics["shares"],
            likes=total_metrics["likes"],
            comments=total_metrics["comments"],
            saves=total_metrics["saves"],
            watch_time=total_metrics["watch_time"],
            retention_rate=sum(m["retention"] for m in raw_metrics) / len(raw_metrics),
            follower_growth=total_metrics["followers_delta"],
            sentiment_score=sum(m["sentiment"] for m in raw_metrics) / len(raw_metrics),
            platform_metrics=platform_metrics
        )

    def analyze_trends(
        self,
        snapshots: List[MetricsSnapshot],
        window_size: int = 24
    ) -> Dict[str, float]:
        """Analyze trends in metrics over the specified window."""
        if len(snapshots) < 2:
            return {}
            
        sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp)
        recent = sorted_snapshots[-window_size:]
        
        engagement_trend = np.polyfit(
            range(len(recent)),
            [s.engagement_rate for s in recent],
            1
        )[0]
        
        viral_trend = np.polyfit(
            range(len(recent)),
            [s.viral_coefficient for s in recent],
            1
        )[0]
        
        reach_growth = (
            recent[-1].reach - recent[0].reach
        ) / recent[0].reach if recent[0].reach > 0 else 0
        
        return {
            "engagement_trend": engagement_trend,
            "viral_trend": viral_trend,
            "reach_growth": reach_growth,
            "follower_growth_rate": sum(s.follower_growth for s in recent) / len(recent)
        }

    async def start_monitoring(
        self,
        interval: int = 300,
        callback: Optional[callable] = None
    ):
        """Start real-time metrics monitoring."""
        self.monitoring = True
        while self.monitoring:
            try:
                snapshot = await self.create_snapshot()
                self._persist_snapshot(snapshot)
                
                if callback:
                    await callback(snapshot)
                    
                # Check thresholds and raise alerts
                alerts = self._check_thresholds(snapshot)
                if alerts:
                    await self._handle_alerts(alerts)
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Error in metrics monitoring: {e}")
                await asyncio.sleep(interval)

    def stop_monitoring(self):
        """Stop real-time metrics monitoring."""
        self.monitoring = False

    def _persist_snapshot(self, snapshot: MetricsSnapshot):
        """Persist metrics snapshot to storage."""
        date_path = self.storage_path / snapshot.timestamp.strftime("%Y-%m-%d")
        date_path.mkdir(exist_ok=True)
        
        file_path = date_path / f"{snapshot.timestamp.strftime('%H%M%S')}.json"
        with open(file_path, 'w') as f:
            json.dump(snapshot.__dict__, f)
            
        # Cache for quick access
        cache_key = f"metrics:{snapshot.timestamp.strftime('%Y%m%d%H%M%S')}"
        self.cache.set(cache_key, snapshot)

    def _check_thresholds(
        self,
        snapshot: MetricsSnapshot
    ) -> List[Tuple[str, float, float]]:
        """Check metrics against defined thresholds."""
        alerts = []
        
        if snapshot.engagement_rate < self.alert_thresholds.engagement_rate_min:
            alerts.append(
                ("engagement_rate", snapshot.engagement_rate,
                self.alert_thresholds.engagement_rate_min)
            )
            
        if snapshot.viral_coefficient < self.alert_thresholds.viral_coefficient_min:
            alerts.append(
                ("viral_coefficient", snapshot.viral_coefficient,
                self.alert_thresholds.viral_coefficient_min)
            )
            
        if snapshot.sentiment_score < self.alert_thresholds.sentiment_score_min:
            alerts.append(
                ("sentiment_score", snapshot.sentiment_score,
                self.alert_thresholds.sentiment_score_min)
            )
            
        if snapshot.retention_rate < self.alert_thresholds.retention_rate_min:
            alerts.append(
                ("retention_rate", snapshot.retention_rate,
                self.alert_thresholds.retention_rate_min)
            )
            
        return alerts

    async def _handle_alerts(
        self,
        alerts: List[Tuple[str, float, float]]
    ):
        """Handle threshold alerts by logging and notifying."""
        for metric, value, threshold in alerts:
            print(
                f"ALERT: {metric} is below threshold. "
                f"Current: {value:.2f}, Threshold: {threshold:.2f}"
            )
            # Here you would typically integrate with a notification system
            # await self.notify_stakeholders(metric, value, threshold)

    async def get_historical_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[MetricsSnapshot]:
        """Retrieve historical metrics within the specified date range."""
        snapshots = []
        current_date = start_date
        
        while current_date <= end_date:
            date_path = self.storage_path / current_date.strftime("%Y-%m-%d")
            if date_path.exists():
                for file_path in date_path.glob("*.json"):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        snapshots.append(MetricsSnapshot(**data))
            
            current_date += timedelta(days=1)
            
        return sorted(snapshots, key=lambda x: x.timestamp)

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Type

from ..platforms.base_platform_client import MetricsSnapshot
from ..platforms.instagram_client import InstagramClient
from ..platforms.tiktok_client import TikTokClient
from ..platforms.youtube_client import YouTubeClient

class MetricsAggregator:
    """Aggregates metrics from multiple platform clients"""
    
    def __init__(self, platform_configs: Dict[str, dict]):
        self.platform_configs = platform_configs
        self.clients = {}
        self._setup_clients()
        
    def _setup_clients(self):
        """Initialize platform-specific clients"""
        client_classes = {
            'instagram': InstagramClient,
            'tiktok': TikTokClient,
            'youtube': YouTubeClient
        }
        
        for platform, config in self.platform_configs.items():
            if platform in client_classes:
                self.clients[platform] = client_classes[platform](**config)
    
    async def _collect_platform_metrics(
        self, 
        platform: str,
        client: Type
    ) -> Dict[str, float]:
        """Collect metrics from a specific platform"""
        try:
            return await client.get_metrics()
        except Exception as e:
            print(f"Error collecting metrics from {platform}: {e}")
            return {}
            
    async def collect_metrics(self) -> MetricsSnapshot:
        """Collect and aggregate metrics from all platforms"""
        # Collect metrics from all platforms concurrently
        tasks = [
            self._collect_platform_metrics(platform, client)
            for platform, client in self.clients.items()
        ]
        
        platform_results = await asyncio.gather(*tasks, return_exceptions=True)
        platform_metrics = {
            platform: metrics for platform, metrics in 
            zip(self.clients.keys(), platform_results)
            if isinstance(metrics, dict)
        }
        
        # Aggregate metrics across platforms
        total_audience = sum(
            metrics.get('audience_size', 0) 
            for metrics in platform_metrics.values()
        )
        total_engagements = sum(
            metrics.get('total_engagements', 0) 
            for metrics in platform_metrics.values()
        )
        
        # Calculate aggregate engagement rate
        engagement_rate = (
            total_engagements / total_audience 
            if total_audience > 0 else 0.0
        )
        
        # Calculate viral metrics
        total_reach = sum(
            metrics.get('content_reach', 0) 
            for metrics in platform_metrics.values()
        )
        viral_reach = sum(
            metrics.get('viral_reach', 0) 
            for metrics in platform_metrics.values()
        )
        
        viral_coefficient = (
            viral_reach / total_reach 
            if total_reach > 0 else 0.0
        )
        
        return MetricsSnapshot(
            # Growth metrics
            audience_size=total_audience,
            audience_growth_rate=sum(
                metrics.get('audience_growth_rate', 0)
                for metrics in platform_metrics.values()
            ) / len(platform_metrics) if platform_metrics else 0.0,
            new_followers=sum(
                metrics.get('new_followers', 0)
                for metrics in platform_metrics.values()
            ),
            unfollows=sum(
                metrics.get('unfollows', 0)
                for metrics in platform_metrics.values()
            ),
            
            # Engagement metrics
            engagement_rate=engagement_rate,
            total_engagements=total_engagements,
            comments_count=sum(
                metrics.get('comments_count', 0)
                for metrics in platform_metrics.values()
            ),
            shares_count

