from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from starlette.websockets import WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse
import pandas as pd
from datetime import timedelta
import asyncio
import json

from .auth import (
    Token, User, authenticate_user, create_access_token,
    get_current_active_user, check_admin_role, check_analyst_role,
    ACCESS_TOKEN_EXPIRE_MINUTES, check_rate_limit
)
from ...ml.predictor import MLPredictor
from ...metrics.aggregator import MetricsAggregator
from ...metrics.types import (
    MetricsSnapshot, PlatformMetrics, ViralScore,
    TrendAnalysis, OptimizationSuggestion
)

app = FastAPI(
    title="Viral Marketing Analytics API",
    description="API for viral marketing metrics, predictions, and optimization",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
metrics_aggregator = MetricsAggregator()
ml_predictor = MLPredictor()

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

# Metrics endpoints
@app.get("/metrics/current", response_model=MetricsSnapshot)
async def get_current_metrics(
    current_user: User = Depends(get_current_active_user)
) -> MetricsSnapshot:
    await check_rate_limit(current_user)
    return await metrics_aggregator.get_current_metrics()

@app.get("/metrics/historical")
async def get_historical_metrics(
    start_date: datetime,
    end_date: datetime,
    platform: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> List[MetricsSnapshot]:
    await check_rate_limit(current_user)
    return await metrics_aggregator.get_historical_metrics(
        start_date, end_date, platform
    )

@app.get("/metrics/export")
async def export_metrics(
    start_date: datetime,
    end_date: datetime,
    format: str = Query(..., regex="^(csv|json|excel)$"),
    current_user: User = Depends(get_current_active_user)
) -> StreamingResponse:
    await check_rate_limit(current_user)
    check_analyst_role(current_user)
    
    metrics = await metrics_aggregator.get_historical_metrics(start_date, end_date)
    df = pd.DataFrame([metric.dict() for metric in metrics])
    
    if format == "csv":
        return StreamingResponse(
            iter([df.to_csv(index=False)]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=metrics.csv"}
        )
    elif format == "json":
        return StreamingResponse(
            iter([df.to_json(orient="records")]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=metrics.json"}
        )
    else:  # excel
        return StreamingResponse(
            iter([df.to_excel(index=False)]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=metrics.xlsx"}
        )

# ML Prediction endpoints
@app.get("/predict/viral-potential", response_model=ViralScore)
async def predict_viral_potential(
    content_url: str,
    platform: str,
    current_user: User = Depends(get_current_active_user)
) -> ViralScore:
    await check_rate_limit(current_user)
    check_analyst_role(current_user)
    return await ml_predictor.predict_viral_potential(content_url, platform)

@app.get("/predict/trends", response_model=TrendAnalysis)
async def analyze_trends(
    platform: str,
    timeframe: str = Query(..., regex="^(day|week|month)$"),
    current_user: User = Depends(get_current_active_user)
) -> TrendAnalysis:
    await check_rate_limit(current_user)
    check_analyst_role(current_user)
    return await ml_predictor.analyze_trends(platform, timeframe)

@app.get("/optimize/content", response_model=List[OptimizationSuggestion])
async def get_optimization_suggestions(
    content_url: str,
    platform: str,
    current_user: User = Depends(get_current_active_user)
) -> List[OptimizationSuggestion]:
    await check_rate_limit(current_user)
    check_analyst_role(current_user)
    return await ml_predictor.get_optimization_suggestions(content_url, platform)

# Real-time metrics streaming
@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            metrics = await metrics_aggregator.get_current_metrics()
            await manager.broadcast({"metrics": metrics.dict()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/metrics/stream")
async def stream_metrics(
    current_user: User = Depends(get_current_active_user)
) -> EventSourceResponse:
    async def event_generator():
        while True:
            metrics = await metrics_aggregator.get_current_metrics()
            yield {
                "event": "metrics",
                "data": json.dumps(metrics.dict())
            }
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"detail": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {"detail": "Internal server error", "status_code": 500}

# Health check
@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

