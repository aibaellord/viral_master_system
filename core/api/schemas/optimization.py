from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, constr
from enum import Enum

class OptimizationType(str, Enum):
    CONTENT = "content"
    STRATEGY = "strategy"
    PERFORMANCE = "performance"
    VIRAL = "viral"
    ENGAGEMENT = "engagement"

class MetricType(str, Enum):
    VIRAL_COEFFICIENT = "viral_coefficient"
    ENGAGEMENT_RATE = "engagement_rate"
    CONVERSION_RATE = "conversion_rate"
    SHARE_RATE = "share_rate"
    NETWORK_EFFECT = "network_effect"

class PlatformMetrics(BaseModel):
    platform_id: str
    viral_coefficient: float = Field(..., description="Current viral coefficient")
    engagement_rate: float = Field(..., description="Current engagement rate")
    reach_multiplier: float = Field(..., description="Network reach multiplier")
    conversion_rate: float = Field(..., description="Content conversion rate")
    share_triggers: Dict[str, float] = Field(..., description="Share trigger effectiveness")

class OptimizationGoals(BaseModel):
    target_viral_coefficient: float = Field(..., ge=1.0, description="Target viral coefficient")
    min_engagement_rate: float = Field(..., ge=0.0, le=1.0, description="Minimum engagement rate")
    target_conversion_rate: float = Field(..., ge=0.0, le=1.0, description="Target conversion rate")
    cost_per_action: float = Field(..., ge=0.0, description="Maximum cost per action")
    roi_target: float = Field(..., ge=1.0, description="Minimum ROI target")

class ABTestVariant(BaseModel):
    variant_id: str
    content_params: Dict[str, any]
    platform_settings: Dict[str, Dict]
    targeting_rules: Optional[Dict[str, any]]

class OptimizationResult(BaseModel):
    entity_id: str
    optimization_type: OptimizationType
    start_time: datetime
    end_time: datetime
    initial_metrics: PlatformMetrics
    final_metrics: PlatformMetrics
    improvements: Dict[str, float]
    recommendations: List[str]
    next_actions: List[Dict[str, any]]

class PerformanceMetrics(BaseModel):
    viral_score: float = Field(..., ge=0.0, le=1.0)
    engagement_quality: float = Field(..., ge=0.0, le=1.0)
    network_reach: float = Field(..., ge=1.0)
    conversion_impact: float = Field(..., ge=0.0)
    cost_efficiency: float = Field(..., ge=0.0)

class OptimizationConfig(BaseModel):
    auto_optimization: bool = True
    update_frequency: int = Field(..., ge=60, description="Update frequency in seconds")
    performance_thresholds: Dict[str, float]
    alert_conditions: Dict[str, Dict[str, float]]
    platform_weights: Dict[str, float]
    optimization_constraints: Dict[str, any]

class ROIMetrics(BaseModel):
    total_cost: float
    total_revenue: float
    roi_ratio: float
    optimization_cost_breakdown: Dict[str, float]
    revenue_sources: Dict[str, float]
    improvement_metrics: Dict[str, float]
    projected_growth: Dict[str, float]

class OptimizationHistory(BaseModel):
    entity_id: str
    timestamp: datetime
    optimization_type: OptimizationType
    metrics_before: PlatformMetrics
    metrics_after: PlatformMetrics
    actions_taken: List[Dict[str, any]]
    impact_score: float = Field(..., ge=0.0, le=1.0)

class OptimizationPlan(BaseModel):
    entity_id: str
    optimization_steps: List[Dict[str, any]]
    expected_improvements: Dict[str, float]
    resource_requirements: Dict[str, any]
    timeline: Dict[str, datetime]
    dependencies: List[str]
    risk_assessment: Dict[str, float]

class ViralOptimizationMetrics(BaseModel):
    viral_coefficient: float = Field(..., ge=0.0)
    network_growth_rate: float = Field(..., ge=0.0)
    share_velocity: float = Field(..., ge=0.0)
    audience_reach: float = Field(..., ge=0.0)
    viral_cycle_time: float = Field(..., ge=0.0)
    k_factor: float = Field(..., ge=0.0)
    retention_rate: float = Field(..., ge=0.0, le=1.0)

class OptimizationAlert(BaseModel):
    alert_id: str
    entity_id: str
    alert_type: str
    severity: str
    timestamp: datetime
    metric_value: float
    threshold_value: float
    recommendation: str
    auto_resolution: Optional[Dict[str, any]]

