from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union
from uuid import UUID

class PlatformType(Enum):
    INSTAGRAM = auto()
    TIKTOK = auto()
    YOUTUBE = auto()
    UNKNOWN = auto()

class MetricCategory(Enum):
    ENGAGEMENT = auto()
    REACH = auto()
    CONVERSION = auto()
    RETENTION = auto()
    MONETIZATION = auto()
    GROWTH = auto()

@dataclass
class BaseMetric:
    """Base class for all metrics with common fields."""
    timestamp: datetime
    platform: PlatformType
    source_id: str
    metric_id: UUID
    category: MetricCategory
    confidence_score: float = 1.0
    is_validated: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class EngagementMetrics(BaseMetric):
    """Engagement-specific metrics across platforms."""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    click_through_rate: float = 0.0
    average_watch_time: float = 0.0
    completion_rate: float = 0.0
    interaction_rate: float = 0.0

@dataclass
class ReachMetrics(BaseMetric):
    """Reach and impression metrics."""
    impressions: int = 0
    unique_viewers: int = 0
    total_reach: int = 0
    organic_reach: int = 0
    paid_reach: int = 0
    viral_reach: int = 0

@dataclass
class TimeSeries:
    """Time series data for tracking metric changes."""
    metric_type: str
    data_points: List[tuple[datetime, float]]
    interval: str  # e.g., '1h', '1d', '1w'
    start_time: datetime
    end_time: datetime
    is_normalized: bool = False

@dataclass
class AggregationResult:
    """Results from metric aggregation operations."""
    metric_category: MetricCategory
    period_start: datetime
    period_end: datetime
    mean: float
    median: float
    max: float
    min: float
    std_dev: float
    sample_size: int
    confidence_interval: tuple[float, float]

@dataclass
class AnalyticsResult:
    """Comprehensive analytics results including trends and predictions."""
    metric_data: Union[EngagementMetrics, ReachMetrics]
    time_series: TimeSeries
    trend_coefficient: float
    prediction_horizon: datetime
    predicted_values: List[float]
    confidence_intervals: List[tuple[float, float]]
    anomalies: List[datetime]
    change_points: List[datetime]
    seasonality_factors: Dict[str, float]

@dataclass
class Alert:
    """Alert configuration and status."""
    alert_id: UUID
    metric_category: MetricCategory
    threshold: float
    condition: str  # e.g., '>', '<', '>=', '<='
    window_size: str  # e.g., '1h', '1d'
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    notification_channels: List[str] = field(default_factory=list)
    cooldown_period: str = '1h'
    severity: str = 'medium'

@dataclass
class Report:
    """Structured report containing multiple metric analyses."""
    report_id: UUID
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    metrics: List[Union[EngagementMetrics, ReachMetrics]]
    analytics: List[AnalyticsResult]
    aggregations: List[AggregationResult]
    alerts: List[Alert]
    summary: str
    recommendations: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)

# Platform-specific metric implementations
@dataclass
class InstagramMetrics(EngagementMetrics):
    """Instagram-specific metrics."""
    story_replies: int = 0
    story_exits: int = 0
    profile_visits: int = 0
    website_clicks: int = 0
    reel_plays: int = 0
    reel_completion_rate: float = 0.0

@dataclass
class TikTokMetrics(EngagementMetrics):
    """TikTok-specific metrics."""
    video_views: int = 0
    follow_rate: float = 0.0
    share_rate: float = 0.0
    comment_rate: float = 0.0
    duets: int = 0
    stitches: int = 0
    live_engagement: float = 0.0

@dataclass
class YouTubeMetrics(EngagementMetrics):
    """YouTube-specific metrics."""
    subscribers_gained: int = 0
    subscribers_lost: int = 0
    average_view_duration: float = 0.0
    playlist_adds: int = 0
    card_clicks: int = 0
    end_screen_clicks: int = 0
    revenue: float = 0.0
    cpm: float = 0.0

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class RetentionData:
    """Represents video retention data over time intervals."""
    intervals: List[str]  # Time intervals (e.g., "0-30", "31-60")
    viewers: List[int]    # Number of viewers for each interval
    retention_rates: List[float]  # Retention rate as percentage for each interval

@dataclass
class SubscriptionMetrics:
    """Represents channel subscription and engagement metrics."""
    total_subscribers: int
    total_views: int
    total_videos: int
    growth_rate: Optional[float] = None  # Percentage growth over previous perio

