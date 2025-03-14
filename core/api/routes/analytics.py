from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio

from core.auth import get_current_user, RateLimiter
from core.metrics import MetricsAggregator
from core.ml.predictor import MLPredictor
from core.automation.viral_engine import ViralLoopEngine
from core.analytics import AnalyticsEngine
from core.schemas.analytics import (
    MetricsResponse,
    NetworkAnalysis,
    PerformanceMetrics,
    PredictiveInsights,
    ReportConfig,
    AlertConfig,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])
rate_limiter = RateLimiter(requests_per_minute=60)

@router.get("/metrics/realtime", response_model=MetricsResponse)
async def get_realtime_metrics(
    platform: Optional[str] = None,
    metrics: List[str] = Query(None),
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Get real-time metrics across all or specific platforms.
    Supports metric filtering and real-time updates.
    """
    try:
        aggregator = MetricsAggregator()
        metrics_data = await aggregator.get_realtime_metrics(
            user_id=user.id,
            platform=platform,
            metrics=metrics
        )
        return MetricsResponse(metrics=metrics_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/historical")
async def get_historical_metrics(
    start_date: datetime,
    end_date: datetime,
    interval: str = "1h",
    platform: Optional[str] = None,
    metrics: List[str] = Query(None),
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Retrieve historical metrics with customizable time ranges and intervals.
    Supports aggregation and trend analysis.
    """
    try:
        analytics = AnalyticsEngine()
        historical_data = await analytics.get_historical_metrics(
            user_id=user.id,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            platform=platform,
            metrics=metrics
        )
        return historical_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network", response_model=NetworkAnalysis)
async def analyze_network(
    timeframe: str = "24h",
    platform: Optional[str] = None,
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Analyze viral spread patterns and network effects.
    Includes influence tracking and engagement analysis.
    """
    try:
        viral_engine = ViralLoopEngine()
        network_data = await viral_engine.analyze_network_effects(
            user_id=user.id,
            timeframe=timeframe,
            platform=platform
        )
        return NetworkAnalysis(**network_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=PerformanceMetrics)
async def monitor_performance(
    metrics: List[str] = Query(None),
    interval: str = "5m",
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Monitor system performance and health metrics.
    Supports real-time updates and benchmarking.
    """
    try:
        analytics = AnalyticsEngine()
        performance_data = await analytics.get_performance_metrics(
            user_id=user.id,
            metrics=metrics,
            interval=interval
        )
        return PerformanceMetrics(**performance_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights", response_model=PredictiveInsights)
async def get_ai_insights(
    timeframe: str = "7d",
    platform: Optional[str] = None,
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Get AI-driven insights and predictions.
    Includes pattern recognition and anomaly detection.
    """
    try:
        predictor = MLPredictor()
        insights = await predictor.generate_insights(
            user_id=user.id,
            timeframe=timeframe,
            platform=platform
        )
        return PredictiveInsights(**insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts")
async def configure_alerts(
    config: AlertConfig,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Configure performance and metric alerts.
    Supports custom thresholds and notification channels.
    """
    try:
        analytics = AnalyticsEngine()
        alert_id = await analytics.configure_alerts(
            user_id=user.id,
            config=config
        )
        background_tasks.add_task(
            analytics.start_alert_monitoring,
            alert_id=alert_id
        )
        return {"alert_id": alert_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reports")
async def generate_report(
    config: ReportConfig,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Generate custom analytics reports.
    Supports multiple formats and scheduled generation.
    """
    try:
        analytics = AnalyticsEngine()
        report_id = await analytics.schedule_report(
            user_id=user.id,
            config=config
        )
        if config.schedule:
            background_tasks.add_task(
                analytics.setup_scheduled_report,
                report_id=report_id
            )
        else:
            background_tasks.add_task(
                analytics.generate_report,
                report_id=report_id
            )
        return {"report_id": report_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def stream_metrics(
    metrics: List[str] = Query(None),
    platform: Optional[str] = None,
    user = Depends(get_current_user),
    rate_limit = Depends(rate_limiter)
):
    """
    Stream real-time metrics updates.
    Supports filtering and platform-specific streams.
    """
    try:
        aggregator = MetricsAggregator()
        return StreamingResponse(
            aggregator.stream_metrics(
                user_id=user.id,
                metrics=metrics,
                platform=platform
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
import logging

from core.api.schemas import (
    AnalyticsRequest,
    PlatformType,
    ViralMetrics,
    ErrorResponse
)
from core.api.auth import get_current_user
from core.automation.viral_engine import ViralLoopEngine
from core.analytics.analyzer import AnalyticsEngine
from core.monitoring.reporter import MetricsReporter

# Configure logging
logger

