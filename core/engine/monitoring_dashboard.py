import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import websockets
import json
from dataclasses import dataclass
from enum import Enum

from .viral_trigger_engine import ViralTriggerEngine
from .viral_analytics import ViralAnalytics

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float

class MonitoringDashboard:
    def __init__(self, viral_engine: ViralTriggerEngine, analytics: ViralAnalytics):
        self.viral_engine = viral_engine
        self.analytics = analytics
        self.metrics: Dict[str, float] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alerts: List[Alert] = []
        self.websocket_clients = set()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default thresholds
        self._init_default_thresholds()
        
    def _init_default_thresholds(self):
        """Initialize default monitoring thresholds."""
        self.thresholds = {
            "viral_coefficient": {
                "warning": 0.8,
                "critical": 0.5
            },
            "engagement_rate": {
                "warning": 2.0,
                "critical": 1.0
            },
            "conversion_rate": {
                "warning": 1.5,
                "critical": 0.8
            }
        }
    
    async def start_monitoring(self):
        """Start the real-time monitoring system."""
        try:
            await asyncio.gather(
                self._start_websocket_server(),
                self._start_metrics_collection(),
                self._start_automated_reporting()
            )
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            raise
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        server = await websockets.serve(self._handle_websocket_connection, "localhost", 8765)
        await server.wait_closed()
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle individual WebSocket connections."""
        try:
            self.websocket_clients.add(websocket)
            while True:
                await self._send_metrics_update(websocket)
                await asyncio.sleep(1)
        finally:
            self.websocket_clients.remove(websocket)
    
    async def _start_metrics_collection(self):
        """Collect metrics from ViralTriggerEngine and ViralAnalytics."""
        while True:
            try:
                # Collect metrics from viral engine
                self.metrics.update({
                    "viral_coefficient": await self.viral_engine.get_viral_coefficient(),
                    "engagement_rate": await self.analytics.get_engagement_rate(),
                    "conversion_rate": await self.analytics.get_conversion_rate(),
                    "platform_performance": await self.viral_engine.get_platform_performance()
                })
                
                # Check thresholds and generate alerts
                await self._check_thresholds()
                
                # Broadcast updates to all connected clients
                await self._broadcast_metrics()
                
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(10)  # Retry after 10 seconds
    
    async def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts."""
        for metric_name, value in self.metrics.items():
            if metric_name in self.thresholds:
                thresholds = self.thresholds[metric_name]
                
                if value < thresholds["critical"]:
                    await self._generate_alert(AlertLevel.CRITICAL, metric_name, value, thresholds["critical"])
                elif value < thresholds["warning"]:
                    await self._generate_alert(AlertLevel.WARNING, metric_name, value, thresholds["warning"])
    
    async def _generate_alert(self, level: AlertLevel, metric_name: str, current_value: float, threshold_value: float):
        """Generate and store alerts."""
        alert = Alert(
            level=level,
            message=f"{metric_name} has fallen below {level.value} threshold",
            timestamp=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value
        )
        self.alerts.append(alert)
        await self._broadcast_alert(alert)
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all connected clients."""
        message = {
            "type": "metrics_update",
            "data": self.metrics,
            "timestamp": datetime.now().isoformat()
        }
        await self._broadcast_message(message)
    
    async def _broadcast_alert(self, alert: Alert):
        """Broadcast alerts to all connected clients."""
        message = {
            "type": "alert",
            "data": {
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value
            }
        }
        await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if self.websocket_clients:
            websockets_coro = [client.send(json.dumps(message))
                            for client in self.websocket_clients]
            await asyncio.gather(*websockets_coro, return_exceptions=True)
    
    async def _start_automated_reporting(self):
        """Generate and distribute automated reports."""
        while True:
            try:
                report = await self._generate_report()
                await self._distribute_report(report)
                await asyncio.sleep(3600)  # Generate report every hour
            except Exception as e:
                self.logger.error(f"Error generating report: {str(e)}")
                await asyncio.sleep(300)  # Retry after 5 minutes
    
    async def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": self.metrics,
            "alerts_summary": [vars(alert) for alert in self.alerts[-10:]],
            "optimization_suggestions": await self._generate_optimization_suggestions(),
            "platform_comparisons": await self._generate_platform_comparisons()
        }
    
    async def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on current metrics."""
        suggestions = []
        try:
            if self.metrics["viral_coefficient"] < 1.0:
                suggestions.append("Consider increasing social sharing incentives")
            if self.metrics["engagement_rate"] < 2.0:
                suggestions.append("Analyze content timing and format for better engagement")
            # Add more suggestion logic here
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
        return suggestions
    
    async def _generate_platform_comparisons(self) -> Dict[str, Any]:
        """Generate platform-specific performance comparisons."""
        return await self.viral_engine.get_platform_performance()
    
    async def _distribute_report(self, report: Dict[str, Any]):
        """Distribute generated report to relevant stakeholders."""
        message = {
            "type": "report",
            "data": report
        }
        await self._broadcast_message(message)
    
    def update_threshold(self, metric_name: str, warning: float, critical: float):
        """Update monitoring thresholds for a specific metric."""
        if metric_name not in self.thresholds:
            raise ValueError(f"Unknown metric: {metric_name}")
        self.thresholds[metric_name] = {
            "warning": warning,
            "critical": critical
        }

