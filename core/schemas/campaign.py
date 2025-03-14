from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, conint, confloat

class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    OPTIMIZING = "optimizing"

class OptimizationType(str, Enum):
    VIRAL_GROWTH = "viral_growth"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    ROI = "roi"
    CROSS_PLATFORM = "cross_platform"

class ContentType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    CAROUSEL = "carousel"
    STORY = "story"
    REELS = "reels"
    LIVE = "live"

class PlatformType(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    TWITTER = "twitter"

# Campaign Models
class CampaignCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    goals: List[str] = Field(..., min_items=1)
    target_audience: Dict[str, Union[str, List[str]]]
    platforms: List[PlatformType]
    budget: float = Field(..., gt=0)
    start_date: datetime
    end_date: Optional[datetime]
    optimization_type: OptimizationType
    viral_coefficients: Dict[str, float] = Field(
        ...,
        description="Target viral coefficients for different platforms"
    )

class Campaign(CampaignCreate):
    id: str
    status: CampaignStatus
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Union[str, float, datetime]]]

class CampaignUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str]
    goals: Optional[List[str]]
    target_audience: Optional[Dict[str, Union[str, List[str]]]]
    budget: Optional[float] = Field(None, gt=0)
    end_date: Optional[datetime]
    optimization_type: Optional[OptimizationType]
    viral_coefficients: Optional[Dict[str, float]]

class CampaignOptimization(BaseModel):
    campaign_id: str
    optimization_type: OptimizationType
    parameters: Dict[str, Union[str, float, List[str]]]
    constraints: Optional[Dict[str, Union[float, List[float]]]]
    target_metrics: Dict[str, float]
    platform_weights: Dict[PlatformType, float]
    auto_optimize: bool = Field(default=False)

class CampaignPerformance(BaseModel):
    campaign_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    viral_coefficients: Dict[str, float]
    platform_performance: Dict[PlatformType, Dict[str, float]]
    growth_rate: float
    roi: float
    prediction_accuracy: float

# Content Models
class ContentCreate(BaseModel):
    campaign_id: str
    content_type: ContentType
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str]
    media_urls: List[HttpUrl]
    platforms: List[PlatformType]
    target_metrics: Dict[str, float]
    optimization_parameters: Dict[str, Union[str, float, List[str]]]
    schedule: Optional[Dict[PlatformType, datetime]]

class ContentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str]
    media_urls: Optional[List[HttpUrl]]
    target_metrics: Optional[Dict[str, float]]
    optimization_parameters: Optional[Dict[str, Union[str, float, List[str]]]]
    schedule: Optional[Dict[PlatformType, datetime]]

class ContentOptimization(BaseModel):
    content_id: str
    optimization_type: OptimizationType
    platform_specific_params: Dict[PlatformType, Dict[str, Union[str, float]]]
    viral_potential_score: float = Field(..., ge=0, le=1)
    optimization_history: List[Dict[str, Union[str, float, datetime]]]
    auto_optimize: bool = Field(default=False)

class ContentSchedule(BaseModel):
    content_id: str
    platform_schedule: Dict[PlatformType, List[datetime]]
    optimization_windows: Dict[PlatformType, List[Dict[str, datetime]]]
    cross_platform_timing: Dict[str, List[PlatformType]]
    performance_based_adjustments: bool = Field(default=True)

class ContentPerformance(BaseModel):
    content_id: str
    timestamp: datetime
    platform_metrics: Dict[PlatformType, Dict[str, float]]
    viral_metrics: Dict[str, float]
    engagement_rates: Dict[PlatformType, float]
    share_rates: Dict[PlatformType, float]
    viral_coefficient: float
    predicted_growth: Dict[str, float]

# Platform Models
class PlatformConfig(BaseModel):
    platform: PlatformType
    api_credentials: Dict[str, str]
    post_frequency: Dict[str, int]
    audience_targeting: Dict[str, Union[str, List[str]]]
    content_requirements: Dict[str, Union[str, List[str]]]
    rate_limits: Dict[str, int]
    optimization_preferences: Dict[str, Union[str, float]]

class PlatformMetrics(BaseModel):
    platform: PlatformType
    timestamp: datetime
    audience_metrics: Dict[str, int]
    engagement_metrics: Dict[str, float]
    viral_metrics: Dict[str, float]
    growth_metrics: Dict[str, float]
    performance_benchmarks: Dict[str, float]

class PlatformSync(BaseModel):
    platform: PlatformType
    last_sync: datetime
    sync_status: str
    sync_metrics: Dict[str, Union[int, float]]
    failed_operations: List[Dict[str, str]]
    sync_queue: List[Dict[str, Union[str, datetime]]]

class PlatformOptimization(BaseModel):
    platform: PlatformType
    optimization_type: OptimizationType
    current_parameters: Dict[str, Union[str, float]]
    target_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Union[str, float, datetime]]]
    auto_optimization_rules: Dict[str, Union[str, float, List[str]]]

# Metrics Models
class ViralMetrics(BaseModel):
    timestamp: datetime
    viral_coefficient: float = Field(..., ge=0)
    viral_cycle_time: float = Field(..., gt=0)
    viral_growth_rate: float
    share_rate: float = Field(..., ge=0, le=1)
    viral_reach: int = Field(..., ge=0)
    network_effects: Dict[str, float]
    viral_loops: List[Dict[str, float]]

class PerformanceMetrics(BaseModel):
    timestamp: datetime
    engagement_rate: float = Field(..., ge=0, le=1)
    conversion_rate: float = Field(..., ge=0, le=1)
    audience_growth: float
    retention_rate: float = Field(..., ge=0, le=1)
    platform_performance: Dict[PlatformType, Dict[str, float]]
    cost_metrics: Dict[str, float]

class OptimizationMetrics(BaseModel):
    timestamp: datetime
    optimization_impact: Dict[str, float]
    optimization_costs: Dict[str, float]
    performance_improvements: Dict[str, float]
    resource_utilization: Dict[str, float]
    optimization_efficiency: float = Field(..., ge=0, le=1)

class ROIMetrics(BaseModel):
    timestamp: datetime
    campaign_roi: float
    platform_roi: Dict[PlatformType, float]
    content_roi: Dict[str, float]
    cost_efficiency: Dict[str, float]
    revenue_metrics: Dict[str, float]
    investment_metrics: Dict[str, float]

class GrowthMetrics(BaseModel):
    timestamp: datetime
    audience_growth_rate: float
    engagement_growth: Dict[str, float]
    viral_growth_metrics: Dict[str, float]
    retention_metrics: Dict[str, float]
    platform_growth: Dict[PlatformType, Dict[str, float]]
    predicted_growth: Dict[str, float]

# Analytics Models
class AnalyticsConfig(BaseModel):
    metrics_collection: Dict[str, bool]
    analysis_frequency: Dict[str, str]
    retention_period: Dict[str, int]
    aggregation_rules: Dict[str, str]
    alert_thresholds: Dict[str, float]
    custom_metrics: Dict[str, Dict[str, str]]

class AnalyticsReport(BaseModel):
    report_id: str
    timestamp: datetime
    time_range: Dict[str, datetime]
    metrics_summary: Dict[str, Union[float, Dict[str, float]]]
    trend_analysis: Dict[str, List[float]]
    insights: List[Dict[str, str]]
    recommendations: List[Dict[str, Union[str, float]]]

class TrendAnalysis(BaseModel):
    timestamp: datetime
    trend_metrics: Dict[str, List[float]]
    seasonal_patterns: Dict[str, List[float]]
    growth_trends: Dict[str, Dict[str, float]]
    correlation_analysis: Dict[str, Dict[str, float]]
    predictive_indicators: Dict[str, float]

class PredictionModel(BaseModel):
    model_id: str
    timestamp: datetime
    predictions: Dict[str, List[float]]
    confidence_intervals: Dict[str, List[float]]
    model_accuracy: Dict[str, float]
    feature_importance: Dict[str, float]
    model_parameters: Dict[str, Union[str, float, List[str]]]

class ImpactAnalysis(BaseModel):
    timestamp: datetime
    impact_metrics: Dict[str, float]
    attribution_analysis: Dict[str, Dict[str, float]]
    cross_platform_impact: Dict[PlatformType, Dict[str, float]]
    roi_impact: Dict[str, float]
    long_term_effects: Dict[str, List[float]]

