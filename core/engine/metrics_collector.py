from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import aiohttp
import numpy as np
from pydantic import BaseModel
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    
class MetricPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
class MetricData(BaseModel):
    name: str
    value: Union[int, float]
    timestamp: datetime
    type: MetricType
    labels: Dict[str, str]
    priority: MetricPriority
    
class Alert(BaseModel):
    title: str
    description: str
    priority: MetricPriority
    timestamp: datetime
    metric_name: str
    threshold: float
    current_value: float
    
class MetricsCollector:
    """Advanced metrics collection and analysis system with ML capabilities"""
    def __init__(self, storage_path: Path = Path("metrics_storage")):
        """Initialize metrics collector with advanced capabilities"""
        self.storage_path = storage_path
        self.metrics_buffer: Dict[str, List[MetricData]] = {}
        self.alerts: List[Alert] = []
        # ML models for analysis
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.trend_analyzer = LinearRegression()
        self.capacity_predictor = Prophet()
        self.scaler = StandardScaler()
        # Real-time processing
        self.metrics_queue = asyncio.Queue()
        self.processing_workers = []
        # Distributed computing
        self.cluster_manager = ClusterManager()
        # Backup/Recovery
        self.backup_manager = BackupManager(storage_path)
        # Performance monitoring
        self.performance_tracker = PerformanceTracker()
        
        # Initialize Prometheus metrics
        self.prom_metrics: Dict[str, Any] = {}
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Start Prometheus HTTP server
        start_http_server(8000)
        
    async def collect_metric(self, metric: MetricData) -> None:
        """Collect and store a single metric."""
        try:
            if metric.name not in self.metrics_buffer:
                self.metrics_buffer[metric.name] = []
            self.metrics_buffer[metric.name].append(metric)
            
            # Update Prometheus metrics
            if metric.name not in self.prom_metrics:
                if metric.type == MetricType.COUNTER:
                    self.prom_metrics[metric.name] = Counter(metric.name, "")
                elif metric.type == MetricType.GAUGE:
                    self.prom_metrics[metric.name] = Gauge(metric.name, "")
                else:
                    self.prom_metrics[metric.name] = Histogram(metric.name, "")
            
            # Update Prometheus metric value
            if metric.type == MetricType.COUNTER:
                self.prom_metrics[metric.name].inc(metric.value)
            else:
                self.prom_metrics[metric.name].set(metric.value)
            
            await self.check_alerts(metric)
            await self.persist_metric(metric)
            
        except Exception as e:
            logger.error(f"Error collecting metric {metric.name}: {str(e)}")
            raise
            
    async def persist_metric(self, metric: MetricData) -> None:
        """Persist metric to storage."""
        try:
            file_path = self.storage_path / f"{metric.name}_{metric.timestamp.date()}.json"
            async with aiofile.async_open(file_path, 'a') as f:
                await f.write(json.dumps(metric.dict()) + '\n')
        except Exception as e:
            logger.error(f"Error persisting metric {metric.name}: {str(e)}")
            
    async def check_alerts(self, metric: MetricData) -> None:
        """Check if metric triggers any alerts."""
        try:
            if len(self.metrics_buffer[metric.name]) > 100:
                values = [m.value for m in self.metrics_buffer[metric.name][-100:]]
                scaled_values = self.scaler.fit_transform(np.array(values).reshape(-1, 1))
                predictions = self.anomaly_detector.fit_predict(scaled_values)
                
                if -1 in predictions:
                    alert = Alert(
                        title=f"Anomaly detected in {metric.name}",
                        description=f"Unusual value detected: {metric.value}",
                        priority=metric.priority,
                        timestamp=datetime.now(),
                        metric_name=metric.name,
                        threshold=0,
                        current_value=metric.value
                    )
                    self.alerts.append(alert)
                    await self.notify_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking alerts for {metric.name}: {str(e)}")
            
    async def notify_alert(self, alert: Alert) -> None:
        """Send alert notification."""
        try:
            # Implement your preferred notification method here
            logger.warning(f"Alert: {alert.title} - {alert.description}")
        except Exception as e:
            logger.error(f"Error sending alert notification: {str(e)}")
            
    def generate_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate a comprehensive metrics report."""
        try:
            report = {
                "timestamp": datetime.now(),
                "period": {
                    "start": start_time,
                    "end": end_time
                },
                "metrics_summary": {},
                "alerts_summary": {},
                "trends": {},
                "recommendations": []
            }
            
            for metric_name, metrics in self.metrics_buffer.items():
                filtered_metrics = [m for m in metrics 
                                if start_time <= m.timestamp <= end_time]
                
                if filtered_metrics:
                    values = [m.value for m in filtered_metrics]
                    report["metrics_summary"][metric_name] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values)
                    }
                    
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
            
    def generate_visualization(self, metric_name: str, start_time: datetime, 
                            end_time: datetime) -> go.Figure:
        """Generate interactive visualization for a metric."""
        try:
            filtered_metrics = [m for m in self.metrics_buffer[metric_name]
                            if start_time <= m.timestamp <= end_time]
            
            timestamps = [m.timestamp for m in filtered_metrics]
            values = [m.value for m in filtered_metrics]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=values, name=metric_name))
            fig.update_layout(title=f"{metric_name} Over Time",
                            xaxis_title="Timestamp",
                            yaxis_title="Value")
            return fig
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            raise
            
    async def analyze_trends(self, metric_name: str, window_size: int = 100) -> Dict[str, Any]:
        """Analyze trends and patterns in metric data."""
        try:
            if metric_name not in self.metrics_buffer:
                return {}
                
            metrics = self.metrics_buffer[metric_name][-window_size:]
            values = [m.value for m in metrics]
            
            df = pd.Series(values)
            analysis = {
                "mean": df.mean(),
                "std": df.std(),
                "trend": "increasing" if df.is_monotonic_increasing
                        else "decreasing" if df.is_monotonic_decreasing
                        else "fluctuating",
                "volatility": df.std() / df.mean() if df.mean() != 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            raise
            
    async def cleanup_old_data(self, retention_days: int = 30) -> None:
        """Clean up old metric data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for metric_name in self.metrics_buffer:
                self.metrics_buffer[metric_name] = [
                    m for m in self.metrics_buffer[metric_name]
                    if m.timestamp > cutoff_date
                ]
                
            # Clean up storage files
            for file_path in self.storage_path.glob("*.json"):
                if datetime.strptime(file_path.stem.split("_")[1], "%Y-%m-%d").date() < cutoff_date.date():
                    file_path.unlink()
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            raise

