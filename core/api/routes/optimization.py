from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from core.engine.viral_loop_engine import ViralLoopEngine
from core.automation.optimizer import AutomationOptimizer
from core.ml.predictor import MLPredictor
from core.metrics.aggregator import MetricsAggregator
from core.auth.dependencies import get_current_user
from core.models.optimization import (
    ContentOptimizationRequest,
    ContentOptimizationResponse,
    StrategyRequest,
    StrategyResponse,
    OptimizationMetrics,
    ABTestRequest,
    ABTestResponse,
    OptimizationHistory,
    OptimizationImpact,
    ROIMetrics,
    OptimizationAlert
)

router = APIRouter(
    prefix="/optimization",
    tags=["optimization"],
    dependencies=[Depends(get_current_user)]
)

# Dependency injection
def get_viral_engine():
    return ViralLoopEngine()

def get_optimizer():
    return AutomationOptimizer()

def get_predictor():
    return MLPredictor()

def get_metrics():
    return MetricsAggregator()

@router.post("/content", response_model=ContentOptimizationResponse)
async def optimize_content(
    request: ContentOptimizationRequest,
    background_tasks: BackgroundTasks,
    viral_engine: ViralLoopEngine = Depends(get_viral_engine),
    optimizer: AutomationOptimizer = Depends(get_optimizer),
    predictor: MLPredictor = Depends(get_predictor)
) -> ContentOptimizationResponse:
    """
    Optimize content for maximum viral potential using AI-driven analysis
    """
    try:
        # Initial quick analysis
        viral_potential = await predictor.analyze_viral_potential(request.content)
        
        # Start background optimization
        optimization_task = optimizer.create_optimization_task(request)
        background_tasks.add_task(
            optimizer.optimize_content,
            task_id=optimization_task.id,
            content=request.content,
            platforms=request.target_platforms
        )
        
        return ContentOptimizationResponse(
            task_id=optimization_task.id,
            initial_viral_score=viral_potential.score,
            optimization_suggestions=viral_potential.suggestions,
            estimated_completion=datetime.utcnow() + viral_potential.estimated_duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/content/ab-test", response_model=ABTestResponse)
async def create_ab_test(
    request: ABTestRequest,
    optimizer: AutomationOptimizer = Depends(get_optimizer)
) -> ABTestResponse:
    """
    Create and manage A/B tests for content optimization
    """
    try:
        test = await optimizer.create_ab_test(
            variations=request.variations,
            metrics=request.target_metrics,
            duration=request.duration
        )
        return ABTestResponse(
            test_id=test.id,
            variations=test.variations,
            start_time=test.start_time,
            estimated_completion=test.estimated_completion
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/{content_id}/metrics", response_model=OptimizationMetrics)
async def get_content_metrics(
    content_id: str,
    metrics: MetricsAggregator = Depends(get_metrics)
) -> OptimizationMetrics:
    """
    Get optimization metrics for specific content
    """
    try:
        return await metrics.get_content_metrics(content_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/content/{content_id}/params")
async def update_optimization_params(
    content_id: str,
    params: Dict[str, Any],
    optimizer: AutomationOptimizer = Depends(get_optimizer)
) -> Dict[str, Any]:
    """
    Update optimization parameters for specific content
    """
    try:
        return await optimizer.update_content_params(content_id, params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy", response_model=StrategyResponse)
async def create_strategy(
    request: StrategyRequest,
    viral_engine: ViralLoopEngine = Depends(get_viral_engine)
) -> StrategyResponse:
    """
    Create a new optimization strategy
    """
    try:
        strategy = await viral_engine.create_strategy(
            target_metrics=request.target_metrics,
            platforms=request.platforms,
            constraints=request.constraints
        )
        return StrategyResponse(
            strategy_id=strategy.id,
            recommendations=strategy.recommendations,
            estimated_impact=strategy.estimated_impact
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/{strategy_id}/performance", response_model=OptimizationMetrics)
async def get_strategy_performance(
    strategy_id: str,
    metrics: MetricsAggregator = Depends(get_metrics)
) -> OptimizationMetrics:
    """
    Get performance metrics for a specific strategy
    """
    try:
        return await metrics.get_strategy_metrics(strategy_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/strategy/{strategy_id}")
async def update_strategy(
    strategy_id: str,
    request: StrategyRequest,
    viral_engine: ViralLoopEngine = Depends(get_viral_engine)
) -> StrategyResponse:
    """
    Update an existing optimization strategy
    """
    try:
        return await viral_engine.update_strategy(strategy_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/auto")
async def enable_auto_optimization(
    strategy_id: str,
    optimizer: AutomationOptimizer = Depends(get_optimizer)
) -> Dict[str, Any]:
    """
    Enable automated optimization for a strategy
    """
    try:
        return await optimizer.enable_auto_optimization(strategy_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/viral")
async def optimize_viral_coefficients(
    strategy_id: str,
    viral_engine: ViralLoopEngine = Depends(get_viral_engine)
) -> Dict[str, float]:
    """
    Optimize viral coefficients for maximum spread
    """
    try:
        return await viral_engine.optimize_coefficients(strategy_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/engagement")
async def optimize_engagement(
    content_id: str,
    optimizer: AutomationOptimizer = Depends(get_optimizer)
) -> Dict[str, Any]:
    """
    Optimize engagement metrics for content
    """
    try:
        return await optimizer.optimize_engagement(content_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=List[OptimizationMetrics])
async def get_optimization_metrics(
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    metrics: MetricsAggregator = Depends(get_metrics)
) -> List[OptimizationMetrics]:
    """
    Get optimization metrics for the specified time range
    """
    try:
        return await metrics.get_optimization_metrics(from_date, to_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auto-tune")
async def enable_auto_tuning(
    content_id: str,
    optimizer: AutomationOptimizer = Depends(get_optimizer)
) -> Dict[str, Any]:
    """
    Enable automated parameter tuning for content
    """
    try:
        return await optimizer.enable_auto_tuning(content_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[OptimizationHistory])
async def get_optimization_history(
    limit: int = Query(default=100, le=1000),
    metrics: MetricsAggregator = Depends(get_metrics)
) -> List[OptimizationHistory]:
    """
    Get optimization history
    """
    try:
        return await metrics.get_optimization_history(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/impact", response_model=OptimizationImpact)
async def get_optimization_impact(
    metrics: MetricsAggregator = Depends(get_metrics)
) -> OptimizationImpact:
    """
    Get impact analysis of optimization efforts
    """
    try:
        return await metrics.get_optimization_impact()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/roi", response_model=ROIMetrics)
async def get_roi_metrics(
    metrics: MetricsAggregator = Depends(get_metrics)
) -> ROIMetrics:
    """
    Get ROI metrics for optimization efforts
    """
    try:
        return await metrics.get_roi_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[OptimizationAlert])
async def get_optimization_alerts(
    metrics: MetricsAggregator = Depends(get_metrics)
) -> List[OptimizationAlert]:
    """
    Get optimization alerts and notifications
    """
    try:
        return await metrics.get_optimization_alerts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

