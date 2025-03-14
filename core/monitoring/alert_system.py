from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
import asyncio
import smtplib
from email.message import EmailMessage

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    component: str

class AlertSystem:
    def __init__(self, email_config: Optional[Dict] = None, slack_config: Optional[Dict] = None):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.email_config = email_config
        self.slack_config = slack_config
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage': {
                'warning': 70.0,
                'critical': 85.0
            },
            'memory_usage': {
                'warning': 75.0,
                'critical': 90.0
            },
            'error_rate': {
                'warning': 3.0,
                'critical': 5.0
            },
            'latency': {
                'warning': 1.5,
                'critical': 2.5
            },
            'viral_coefficient': {
                'warning': 0.8,
                'critical': 0.5
            }
        }

    async def check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        new_alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                # Check critical threshold first
                if value >= threshold['critical']:
                    alert = self._create_alert(
                        metric_name,
                        value,
                        threshold['critical'],
                        AlertSeverity.CRITICAL
                    )
                    new_alerts.append(alert)
                    
                # Check warning threshold
                elif value >= threshold['warning']:
                    alert = self._create_alert(
                        metric_name,
                        value,
                        threshold['warning'],
                        AlertSeverity.WARNING
                    )
                    new_alerts.append(alert)
                    
        return new_alerts

    def _create_alert(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        severity: AlertSeverity
    ) -> Alert:
        alert_id = f"{metric_name}_{datetime.now().timestamp()}"
        alert = Alert(
            id=alert_id,
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=self._generate_alert_message(metric_name, value, threshold),
            timestamp=datetime.now(),
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold,
            component=self._get_component_for_metric(metric_name)
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        return alert

    async def notify(self, alert: Alert) -> None:
        # Send notifications based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            await self._send_email_alert(alert)
            await self._send_slack_alert(alert)
        elif alert.severity == AlertSeverity.WARNING:
            await self._send_slack_alert(alert)

    async def _send_email_alert(self, alert: Alert) -> None:
        if not self.email_config:
            return
            
        msg = EmailMessage()
        msg.set_content(self._format_alert_email(alert))
        msg['Subject'] = f"[{alert.severity.value.upper()}] Performance Alert: {alert.metric_name}"
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        
        try:
            with smtplib.SMTP(self.email_config['smtp_server']) as server:
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")

    async def _send_slack_alert(self, alert: Alert) -> None:
        if not self.slack_config:
            return
            
        # Implement Slack notification logic here
        pass

    def _generate_alert_message(self, metric_name: str, value: float, threshold: float) -> str:
        return (
            f"{metric_name.replace('_', ' ').title()} exceeded threshold: "
            f"Current value: {value:.2f}, Threshold: {threshold:.2f}"
        )

    def _get_component_for_metric(self, metric_name: str) -> str:
        component_mapping = {
            'cpu_usage': 'System',
            'memory_usage': 'System',
            'error_rate': 'Application',
            'latency': 'API',
            'viral_coefficient': 'Viral Engine'
        }
        return component_mapping.get(metric_name, 'Unknown')

    def acknowledge_alert(self, alert_id: str) -> None:
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED

    def resolve_alert(self, alert_id: str) -> None:
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.RESOLVED
            del self.active_alerts[alert_id]

    async def cleanup_old_alerts(self, max_age_hours: int = 24) -> None:
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp.timestamp() > cutoff_time
        ]

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        if severity:
            return [
                alert for alert in self.active_alerts.values()
                if alert.severity == severity
            ]
        return list(self.active_alerts.values())

    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        alerts = self.alert_history
        
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return alerts

