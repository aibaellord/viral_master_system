import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from core.engine.viral_orchestrator import ViralOrchestrator
from core.engine.viral_innovation_engine import ViralInnovationEngine
from core.engine.viral_trigger_engine import ViralTriggerEngine
from core.metrics.performance_metrics import MetricsCollector
from core.utils.config import Configuration
from core.utils.exceptions import PipelineError

logger = logging.getLogger(__name__)

@dataclass
class InfluencerMetrics:
    """Tracks key performance metrics for influencers."""
    influencer_id: UUID
    engagement_rate: float = 0.0
    audience_reach: int = 0
    content_performance: Dict[str, float] = field(default_factory=dict)
    viral_coefficient: float = 0.0
    authenticity_score: float = 0.0
    platform_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    roi_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    historical_performance: List[Dict[str, float]] = field(default_factory=list)

    def update_metrics(self, new_data: Dict[str, Union[float, Dict]]) -> None:
        """Updates metrics with new performance data."""
        for key, value in new_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()
        self.historical_performance.append({
            "timestamp": self.last_updated.isoformat(),
            "engagement_rate": self.engagement_rate,
            "viral_coefficient": self.viral_coefficient,
            "authenticity_score": self.authenticity_score
        })

class InfluencerPipeline:
    """Manages influencer analysis, optimization, and performance tracking."""

    def __init__(self, config: Configuration):
        """Initialize the influencer pipeline with configuration."""
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.viral_orchestrator = ViralOrchestrator()
        self.innovation_engine = ViralInnovationEngine()
        self.trigger_engine = ViralTriggerEngine()
        
        self.influencer_metrics: Dict[UUID, InfluencerMetrics] = {}
        self.optimization_model = RandomForestRegressor()
        self.scaler = StandardScaler()
        
        self._initialize_logging()
        self.running = False
        self._initialize_async_tasks()

    def _initialize_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=self.config.get("logging_level", logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _initialize_async_tasks(self) -> None:
        """Initialize async task queues and workers."""
        self.analysis_queue: asyncio.Queue = asyncio.Queue()
        self.optimization_queue: asyncio.Queue = asyncio.Queue()
        self.metrics_update_interval = self.config.get("metrics_update_interval", 300)

    async def start(self) -> None:
        """Start the influencer pipeline processing."""
        try:
            self.running = True
            await asyncio.gather(
                self._run_analysis_worker(),
                self._run_optimization_worker(),
                self._run_metrics_collector()
            )
        except Exception as e:
            logger.error(f"Error starting influencer pipeline: {e}")
            raise PipelineError(f"Failed to start influencer pipeline: {str(e)}")

    async def stop(self) -> None:
        """Stop all pipeline processing."""
        self.running = False
        await self.viral_orchestrator.shutdown()
        logger.info("Influencer pipeline stopped successfully")

    async def analyze_influencer(self, influencer_id: UUID) -> InfluencerMetrics:
        """Analyze influencer performance and impact."""
        try:
            metrics = self.influencer_metrics.get(influencer_id)
            if not metrics:
                metrics = InfluencerMetrics(influencer_id=influencer_id)
                self.influencer_metrics[influencer_id] = metrics

            # Gather performance data from various sources
            platform_data = await self._gather_platform_metrics(influencer_id)
            engagement_data = await self._analyze_engagement_patterns(influencer_id)
            viral_impact = await self.viral_orchestrator.measure_influence(influencer_id)

            # Update metrics with new data
            metrics.update_metrics({
                "platform_metrics": platform_data,
                "engagement_rate": engagement_data["rate"],
                "viral_coefficient": viral_impact["coefficient"],
                "authenticity_score": await self._calculate_authenticity_score(influencer_id)
            })

            await self.metrics_collector.record_metrics(
                "influencer_analysis",
                {str(influencer_id): metrics.__dict__}
            )

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing influencer {influencer_id}: {e}")
            raise PipelineError(f"Influencer analysis failed: {str(e)}")

    async def optimize_performance(self, influencer_id: UUID) -> Dict[str, float]:
        """Optimize influencer performance using AI-driven strategies."""
        try:
            metrics = self.influencer_metrics.get(influencer_id)
            if not metrics:
                raise PipelineError(f"No metrics found for influencer {influencer_id}")

            # Prepare features for optimization
            features = self._prepare_optimization_features(metrics)
            
            # Generate optimization recommendations
            recommendations = await self._generate_recommendations(features)
            
            # Apply optimization strategies
            optimization_results = await self.innovation_engine.apply_strategies(
                influencer_id,
                recommendations
            )

            # Trigger viral mechanisms
            await self.trigger_engine.activate(
                influencer_id,
                optimization_results["trigger_points"]
            )

            return optimization_results

        except Exception as e:
            logger.error(f"Error optimizing performance for {influencer_id}: {e}")
            raise PipelineError(f"Performance optimization failed: {str(e)}")

    async def _run_analysis_worker(self) -> None:
        """Background worker for processing influencer analyses."""
        while self.running:
            try:
                influencer_id = await self.analysis_queue.get()
                await self.analyze_influencer(influencer_id)
                self.analysis_queue.task_done()
            except Exception as e:
                logger.error(f"Analysis worker error: {e}")
                await asyncio.sleep(1)

    async def _run_optimization_worker(self) -> None:
        """Background worker for processing optimization tasks."""
        while self.running:
            try:
                influencer_id = await self.optimization_queue.get()
                await self.optimize_performance(influencer_id)
                self.optimization_queue.task_done()
            except Exception as e:
                logger.error(f"Optimization worker error: {e}")
                await asyncio.sleep(1)

    async def _run_metrics_collector(self) -> None:
        """Periodically collect and update metrics."""
        while self.running:
            try:
                for influencer_id in self.influencer_metrics:
                    await self.metrics_collector.update_metrics(
                        "influencer_pipeline",
                        {str(influencer_id): self.influencer_metrics[influencer_id].__dict__}
                    )
                await asyncio.sleep(self.metrics_update_interval)
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(1)

    async def _gather_platform_metrics(self, influencer_id: UUID) -> Dict[str, Dict[str, float]]:
        """Gather metrics from various social media platforms.
        
        Args:
            influencer_id (UUID): Unique identifier for the influencer

        Returns:
            Dict[str, Dict[str, float]]: Platform-specific metrics including engagement rates,
            follower growth, and content performance metrics.

        Raises:
            PipelineError: If there's an error fetching metrics from any platform
        """
        try:
            platform_metrics = {}
            platforms = self.config.get("supported_platforms", ["instagram", "tiktok", "youtube", "twitter"])
            
            for platform in platforms:
                try:
                    api_client = self._get_platform_api_client(platform)
                    platform_metrics[platform] = await api_client.get_metrics(
                        influencer_id=str(influencer_id),
                        metrics=[
                            "engagement_rate",
                            "follower_count",
                            "post_frequency",
                            "average_likes",
                            "comment_ratio",
                            "share_velocity",
                            "growth_rate"
                        ]
                    )
                    
                    # Normalize metrics to common scale
                    platform_metrics[platform] = {
                        k: float(v) for k, v in platform_metrics[platform].items()
                    }
                    
                    # Add platform-specific metrics
                    if platform == "tiktok":
                        platform_metrics[platform]["viral_coefficient"] = (
                            platform_metrics[platform]["share_velocity"] * 
                            platform_metrics[platform]["engagement_rate"]
                        )
                    elif platform == "youtube":
                        platform_metrics[platform]["watch_time_score"] = await api_client.get_watch_time_metrics(
                            influencer_id=str(influencer_id)
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch metrics for {platform}: {e}")
                    platform_metrics[platform] = {
                        "engagement_rate": 0.0,
                        "follower_count": 0,
                        "error": str(e)
                    }
                    
            return platform_metrics
            
        except Exception as e:
            logger.error(f"Error gathering platform metrics for {influencer_id}: {e}")
            raise PipelineError(f"Failed to gather platform metrics: {str(e)}")

    async def _analyze_engagement_patterns(self, influencer_id: UUID) -> Dict[str, float]:
        """Analyze engagement patterns and trends using historical data.
        
        Args:
            influencer_id (UUID): Unique identifier for the influencer

        Returns:
            Dict[str, float]: Analysis results including engagement rates, trend indicators,
            and temporal patterns.

        Raises:
            PipelineError: If analysis fails
        """
        try:
            # Fetch historical data
            metrics = self.influencer_metrics.get(influencer_id)
            if not metrics or not metrics.historical_performance:
                return {"rate": 0.0, "trend": 0.0, "consistency": 0.0}
            
            # Convert historical data to numpy arrays for analysis
            historical_data = np.array([
                [
                    float(entry["engagement_rate"]),
                    float(entry["viral_coefficient"]),
                    float(entry["authenticity_score"])
                ]
                for entry in metrics.historical_performance
            ])
            
            # Calculate trend indicators
            trend = np.polyfit(
                np.arange(len(historical_data)),
                historical_data[:, 0],  # engagement_rate
                deg=1
            )[0]
            
            # Calculate consistency score
            consistency = 1.0 - np.std(historical_data[:, 0]) / np.mean(historical_data[:, 0])
            
            # Calculate weighted engagement rate
            current_rate = historical_data[-1, 0]
            weighted_rate = (
                current_rate * 0.6 +  # Recent performance
                np.mean(historical_data[:, 0]) * 0.3 +  # Historical average
                (trend * 10) * 0.1  # Trend influence
            )
            
            return {
                "rate": float(weighted_rate),
                "trend": float(trend),
                "consistency": float(consistency),
                "volatility": float(np.std(historical_data[:, 0])),
                "growth_velocity": float(np.mean(np.diff(historical_data[:, 0]))),
                "viral_probability": float(np.mean(historical_data[:, 1])),
                "authenticity_trend": float(np.mean(historical_data[:, 2]))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing engagement patterns for {influencer_id}: {e}")
            raise PipelineError(f"Failed to analyze engagement patterns: {str(e)}")

    async def _calculate_authenticity_score(self, influencer_id: UUID) -> float:
        """Calculate authenticity score using AI analysis of various signals.
        
        Args:
            influencer_id (UUID): Unique identifier for the influencer

        Returns:
            float: Authenticity score between 0 and 1

        Raises:
            PipelineError: If authenticity calculation fails
        """
        try:
            metrics = self.influencer_metrics.get(influencer_id)
            if not metrics:
                return 0.0
                
            # Gather authenticity signals
            signals = await self._gather_authenticity_signals(influencer_id)
            
            # Calculate weighted component scores
            engagement_authenticity = self._analyze_engagement_authenticity(signals["engagement_patterns"])
            content_authenticity = await self._analyze_content_authenticity(signals["content_samples"])
            audience_authenticity = self._analyze_audience_authenticity(signals["audience_metrics"])
            growth_authenticity = self._analyze_growth_patterns(signals["growth_metrics"])
            
            # Calculate final weighted score
            weights = {
                "engagement": 0.3,
                "content": 0.25,
                "audience": 0.25,
                "growth": 0.2
            }
            
            authenticity_score = (
                engagement_authenticity * weights["engagement"] +
                content_authenticity * weights["content"] +
                audience_authenticity * weights["audience"] +
                growth_authenticity * weights["growth"]
            )
            
            # Apply penalty factors
            if signals.get("red_flags", []):
                penalty = len(signals["red_flags"]) * 0.1
                authenticity_score = max(0.0, authenticity_score - penalty)
            
            return float(min(1.0, max(0.0, authenticity_score)))
            
        except Exception as e:
            logger.error(f"Error calculating authenticity score for {influencer_id}: {e}")
            raise PipelineError(f"Failed to calculate authenticity score: {str(e)}")

    def _prepare_optimization_features(self, metrics: InfluencerMetrics) -> np.ndarray:
        """Prepare features for the optimization model."""
        features = np.array([
            metrics.engagement_rate,
            metrics.viral_coefficient,
            metrics.authenticity_score,
            *self._extract_platform_features(metrics.platform_metrics)
        ]).reshape(1, -1)
        return self.scaler.fit_transform(features)

    def _extract_platform_features(self, platform_metrics: Dict[str, Dict[str, float]]) -> List[float]:
        """Extract and normalize relevant features from platform metrics.
        
        Args:
            platform_metrics (Dict[str, Dict[str, float]]): Raw platform metrics

        Returns:
            List[float]: Normalized feature vector for model input

        Raises:
            PipelineError: If feature extraction fails
        """
        try:
            feature_list = []
            
            # Core engagement features
            for platform in self.config.get("supported_platforms", []):
                metrics = platform_metrics.get(platform, {})
                
                # Engagement metrics
                feature_list.extend([
                    metrics.get("engagement_rate", 0.0),
                    metrics.get("comment_ratio", 0.0),
                    metrics.get("share_velocity", 0.0)
                ])
                
                # Growth metrics
                feature_list.extend([
                    metrics.get("follower_count", 0.0),
                    metrics.get("growth_rate", 0.0)
                ])
                
                # Platform-specific features
                if platform == "tiktok":
                    feature_list.append(metrics.get("viral_coefficient", 0.0))
                elif platform == "youtube":
                    feature_list.append(metrics.get("watch_time_score", 0.0))
                else:
                    feature_list.append(0.0)  # Placeholder for platform-specific metric
            
            # Normalize features
            normalized_features = []
            for feature in feature_list:
                try:
                    # Handle potential string values
                    value = float(feature)
                    # Clip extreme values
                    value = min(max(value, -1e6), 1e6)
                    normalized_features.append(value)
                except (ValueError, TypeError):
                    normalized_features.append(0.0)
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting platform features: {e}")
            raise PipelineError(f"Failed to extract platform features: {str(e)}")

    async def _generate_recommendations(self, features: np.ndarray) -> Dict[str, float]:
        """Generate optimization recommendations using the AI model."""
        try:
            predictions = self.optimization_model.predict(features)
            return {
                "content_frequency": float(predictions[0]),
                "engagement_timing": float(predictions[1]),
                "platform_focus": float(predictions[2]),
                "collaboration_score": float(predictions[3])
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise PipelineError(f"Failed to generate recommendations: {str(e)}")

