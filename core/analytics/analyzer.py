from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..ml.predictor import MLPredictor
from ..metrics.metrics_aggregator import MetricsAggregator
from ..metrics.types import MetricsSnapshot, PlatformMetrics
from ..platforms.base_platform_client import BasePlatformClient

@dataclass
class AnalyticsInsight:
    insight_type: str
    description: str
    confidence: float
    timestamp: datetime
    metrics: Dict[str, float]
    recommendations: List[str]

@dataclass
class CampaignPerformance:
    campaign_id: str
    start_date: datetime
    end_date: datetime
    roi: float
    engagement_rate: float
    viral_coefficient: float
    reach: int
    conversion_rate: float
    cost_per_engagement: float
    total_cost: float
    total_revenue: float

class MarketingAnalyzer:
    def __init__(
        self,
        metrics_aggregator: MetricsAggregator,
        ml_predictor: MLPredictor,
        platform_clients: List[BasePlatformClient]
    ):
        self.metrics_aggregator = metrics_aggregator
        self.ml_predictor = ml_predictor
        self.platform_clients = platform_clients
        self.scaler = StandardScaler()
        
    async def analyze_performance(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AnalyticsInsight]:
        """Analyzes performance metrics and generates insights."""
        insights = []
        
        # Get metrics for the period
        metrics = await self.metrics_aggregator.get_metrics_range(start_date, end_date)
        
        # Calculate trend analysis
        trend_insight = self._analyze_trends(metrics)
        insights.append(trend_insight)
        
        # Analyze viral patterns
        viral_insight = self._identify_viral_patterns(metrics)
        insights.append(viral_insight)
        
        # Generate performance benchmarks
        benchmark_insight = await self._calculate_benchmarks(metrics)
        insights.append(benchmark_insight)
        
        # Calculate ROI metrics
        roi_insight = self._calculate_roi_metrics(metrics)
        insights.append(roi_insight)
        
        return insights

    def _analyze_trends(self, metrics: List[MetricsSnapshot]) -> AnalyticsInsight:
        """Performs statistical trend analysis on metrics."""
        engagement_rates = [m.engagement_rate for m in metrics]
        
        # Calculate moving averages
        window = 7
        moving_avg = np.convolve(engagement_rates, np.ones(window)/window, mode='valid')
        
        # Perform trend detection
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(moving_avg)),
            moving_avg
        )
        
        confidence = 1 - p_value
        trend_direction = "upward" if slope > 0 else "downward"
        
        return AnalyticsInsight(
            insight_type="trend_analysis",
            description=f"Engagement shows {trend_direction} trend",
            confidence=confidence,
            timestamp=datetime.now(),
            metrics={"slope": slope, "r_squared": r_value**2},
            recommendations=self._generate_trend_recommendations(slope, confidence)
        )

    def _identify_viral_patterns(self, metrics: List[MetricsSnapshot]) -> AnalyticsInsight:
        """Identifies patterns in viral content performance."""
        # Extract features for clustering
        features = np.array([
            [m.engagement_rate, m.viral_coefficient, m.share_rate]
            for m in metrics
        ])
        
        # Normalize features
        scaled_features = self.scaler.fit_transform(features)
        
        # Perform clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Identify viral cluster (highest centroid values)
        viral_cluster = np.argmax(np.mean(kmeans.cluster_centers_, axis=1))
        viral_patterns = [
            metrics[i] for i in range(len(metrics))
            if clusters[i] == viral_cluster
        ]
        
        return AnalyticsInsight(
            insight_type="viral_patterns",
            description=f"Identified {len(viral_patterns)} viral content patterns",
            confidence=kmeans.score(scaled_features),
            timestamp=datetime.now(),
            metrics={"viral_cluster_size": len(viral_patterns)},
            recommendations=self._generate_viral_recommendations(viral_patterns)
        )

    async def _calculate_benchmarks(
        self,
        metrics: List[MetricsSnapshot]
    ) -> AnalyticsInsight:
        """Calculates performance benchmarks against competitors."""
        competitor_metrics = []
        for client in self.platform_clients:
            competitor_data = await client.get_competitor_metrics()
            competitor_metrics.extend(competitor_data)
            
        # Calculate percentile rankings
        engagement_percentile = stats.percentileofscore(
            [m.engagement_rate for m in competitor_metrics],
            np.mean([m.engagement_rate for m in metrics])
        )
        
        viral_percentile = stats.percentileofscore(
            [m.viral_coefficient for m in competitor_metrics],
            np.mean([m.viral_coefficient for m in metrics])
        )
        
        return AnalyticsInsight(
            insight_type="benchmarks",
            description="Performance benchmark analysis",
            confidence=0.95,
            timestamp=datetime.now(),
            metrics={
                "engagement_percentile": engagement_percentile,
                "viral_percentile": viral_percentile
            },
            recommendations=self._generate_benchmark_recommendations(
                engagement_percentile,
                viral_percentile
            )
        )

    def _calculate_roi_metrics(self, metrics: List[MetricsSnapshot]) -> AnalyticsInsight:
        """Calculates ROI and effectiveness metrics."""
        total_cost = sum(m.campaign_cost for m in metrics)
        total_revenue = sum(m.revenue for m in metrics)
        total_engagement = sum(m.engagement_count for m in metrics)
        
        roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
        cost_per_engagement = total_cost / total_engagement if total_engagement > 0 else 0
        
        return AnalyticsInsight(
            insight_type="roi_analysis",
            description="ROI and cost effectiveness analysis",
            confidence=0.95,
            timestamp=datetime.now(),
            metrics={
                "roi": roi,
                "cost_per_engagement": cost_per_engagement,
                "total_revenue": total_revenue
            },
            recommendations=self._generate_roi_recommendations(roi, cost_per_engagement)
        )

    def _generate_trend_recommendations(self, slope: float, confidence: float) -> List[str]:
        """Generates recommendations based on trend analysis."""
        recommendations = []
        if slope > 0:
            recommendations.extend([
                "Continue current content strategy",
                "Increase posting frequency to capitalize on positive trend",
                "Analyze top performing content for replication"
            ])
        else:
            recommendations.extend([
                "Review and revise content strategy",
                "Analyze audience feedback for improvements",
                "Test new content formats and themes"
            ])
        return recommendations

    def _generate_viral_recommendations(
        self,
        viral_patterns: List[MetricsSnapshot]
    ) -> List[str]:
        """Generates recommendations based on viral pattern analysis."""
        recommendations = []
        if viral_patterns:
            common_features = self._extract_common_features(viral_patterns)
            recommendations.extend([
                f"Focus on {feature} in future content"
                for feature in common_features
            ])
            recommendations.append("Optimize posting timing based on viral content patterns")
        return recommendations

    def _generate_benchmark_recommendations(
        self,
        engagement_percentile: float,
        viral_percentile: float
    ) -> List[str]:
        """Generates recommendations based on benchmark analysis."""
        recommendations = []
        if engagement_percentile < 50:
            recommendations.extend([
                "Improve audience engagement tactics",
                "Study competitor engagement strategies",
                "Increase interactive content"
            ])
        if viral_percentile < 50:
            recommendations.extend([
                "Enhance content shareability",
                "Implement viral triggers in content",
                "Focus on trending topics and formats"
            ])
        return recommendations

    def _generate_roi_recommendations(
        self,
        roi: float,
        cost_per_engagement: float
    ) -> List[str]:
        """Generates recommendations based on ROI analysis."""
        recommendations = []
        if roi < 0.2:  # 20% ROI threshold
            recommendations.extend([
                "Optimize ad spend allocation",
                "Focus on high-converting content formats",
                "Review and adjust campaign targeting"
            ])
        if cost_per_engagement > 0.5:  # $0.50 per engagement threshold
            recommendations.extend([
                "Improve content engagement efficiency",
                "Test different ad formats for better CPE",
                "Refine audience targeting parameters"
            ])
        return recommendations

    def _extract_common_features(
        self,
        viral_patterns: List[MetricsSnapshot]
    ) -> List[str]:
        """Extracts common features from viral content."""
        features = []
        # Analyze content characteristics
        content_types = [p.content_type for p in viral_patterns]
        most_common_type = max(set(content_types), key=content_types.count)
        features.append(most_common_type)
        
        # Analyze posting times
        posting_hours = [p.posted_at.hour for p in viral_patterns]
        peak_hour = max(set(posting_hours), key=posting_hours.count)
        features.append(f"posting at {peak_hour}:00")
        
        return features

