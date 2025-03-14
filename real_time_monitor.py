#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import aiohttp
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RealTimeMonitor")


@dataclass
class PerformanceMetric:
    """A single performance metric with timestamp and value."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    platform: str = "all"
    content_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format for storage/transmission."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "content_id": self.content_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create a PerformanceMetric from dictionary data."""
        return cls(
            name=data["name"],
            value=data["value"],
            timestamp=data.get("timestamp", time.time()),
            platform=data.get("platform", "all"),
            content_id=data.get("content_id", "")
        )


@dataclass
class PerformanceAlert:
    """Alert generated when a metric crosses a threshold."""
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: float = field(default_factory=time.time)
    severity: str = "info"  # info, warning, critical
    message: str = ""
    platform: str = "all"
    content_id: str = ""
    
    def __post_init__(self):
        """Set default message if none provided."""
        if not self.message:
            direction = "above" if self.current_value > self.threshold_value else "below"
            self.message = f"Metric {self.metric_name} is {direction} threshold: {self.current_value} vs {self.threshold_value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format for storage/transmission."""
        return {
            "metric_name": self.metric_name,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "message": self.message,
            "platform": self.platform,
            "content_id": self.content_id
        }


class AdaptiveThreshold:
    """Adaptive threshold that adjusts based on historical data."""
    
    def __init__(
        self, 
        metric_name: str,
        initial_value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        adaptation_rate: float = 0.05,
        window_size: int = 100
    ):
        self.metric_name = metric_name
        self.current_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.history: List[float] = []
    
    def update(self, new_metrics: List[float]) -> None:
        """Update threshold based on new metric values."""
        self.history.extend(new_metrics)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        if not self.history:
            return
            
        # Calculate new threshold based on recent history
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        # Adjust threshold based on mean and standard deviation
        new_threshold = mean + (std * 1.5)  # 1.5 sigma above mean
        
        # Apply adaptation rate to smooth changes
        self.current_value = (
            (1 - self.adaptation_rate) * self.current_value + 
            self.adaptation_rate * new_threshold
        )
        
        # Apply min/max constraints if provided
        if self.min_value is not None:
            self.current_value = max(self.current_value, self.min_value)
        if self.max_value is not None:
            self.current_value = min(self.current_value, self.max_value)
    
    def check(self, value: float) -> bool:
        """Check if value exceeds the current threshold."""
        return value > self.current_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert threshold to dictionary format for storage."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "adaptation_rate": self.adaptation_rate,
            "window_size": self.window_size
        }


class RealTimeMonitor:
    """
    Real-time performance monitor for content metrics across multiple platforms.
    
    Features:
    - Tracks performance metrics in real-time
    - Provides instant feedback for content adjustments
    - Integrates with neural content enhancer
    - Supports multiple platforms
    - Includes performance alerts and notifications
    - Implements adaptive thresholds
    """
    
    def __init__(
        self,
        data_storage_path: str = "./data/metrics",
        alert_handlers: Optional[List[callable]] = None,
        platforms: Optional[List[str]] = None,
        update_interval: float = 5.0,
        neural_enhancer = None  # Type hint omitted for flexibility
    ):
        """
        Initialize the real-time monitor.
        
        Args:
            data_storage_path: Path to store metrics data
            alert_handlers: List of handlers for alert notifications
            platforms: List of platforms to monitor
            update_interval: How often to update metrics (seconds)
            neural_enhancer: Optional neural enhancer component to integrate with
        """
        self.data_storage_path = data_storage_path
        self.alert_handlers = alert_handlers or []
        self.platforms = platforms or ["twitter", "instagram", "facebook", "linkedin", "tiktok", "youtube"]
        self.update_interval = update_interval
        self.neural_enhancer = neural_enhancer
        
        # Ensure data directory exists
        os.makedirs(data_storage_path, exist_ok=True)
        
        # Internal storage
        self.metrics: Dict[str, Dict[str, List[PerformanceMetric]]] = {
            platform: {} for platform in self.platforms
        }
        self.metrics["all"] = {}  # For platform-agnostic metrics
        
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, Dict[str, AdaptiveThreshold]] = {
            platform: {} for platform in self.platforms
        }
        self.thresholds["all"] = {}  # For platform-agnostic thresholds
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Background tasks
        self._update_task = None
        self._running = False
    
    def _initialize_default_thresholds(self) -> None:
        """Initialize default thresholds for common metrics."""
        default_thresholds = {
            "engagement_rate": 0.02,  # 2% engagement rate
            "likes_per_minute": 2.0,
            "shares_per_minute": 0.5,
            "comments_per_minute": 1.0,
            "views_per_minute": 10.0,
            "clicks_per_minute": 3.0,
            "follower_growth_rate": 0.01,  # 1% growth
            "sentiment_score": 0.6,  # 0-1 scale, 0.6 is moderately positive
            "virality_score": 70.0,  # 0-100 scale
        }
        
        # Create thresholds for each platform
        for platform in self.platforms + ["all"]:
            for metric, value in default_thresholds.items():
                # Adjust thresholds based on platform characteristics
                adjusted_value = self._adjust_threshold_for_platform(metric, value, platform)
                
                # Create adaptive threshold
                self.thresholds[platform][metric] = AdaptiveThreshold(
                    metric_name=metric,
                    initial_value=adjusted_value,
                    min_value=adjusted_value * 0.5,  # Allow threshold to drop to 50% of initial
                    adaptation_rate=0.1
                )
    
    def _adjust_threshold_for_platform(self, metric: str, value: float, platform: str) -> float:
        """Adjust threshold values based on platform characteristics."""
        # Platform-specific adjustments
        platform_multipliers = {
            "twitter": {"engagement_rate": 1.2, "shares_per_minute": 1.5},
            "instagram": {"engagement_rate": 1.5, "likes_per_minute": 2.0},
            "tiktok": {"views_per_minute": 5.0, "virality_score": 1.2},
            "youtube": {"views_per_minute": 2.0, "comments_per_minute": 0.7},
            "facebook": {"shares_per_minute": 1.2},
            "linkedin": {"engagement_rate": 0.7, "clicks_per_minute": 1.5},
            "all": {}  # No adjustment for platform-agnostic thresholds
        }
        
        # Apply multiplier if exists for this platform and metric
        multiplier = platform_multipliers.get(platform, {}).get(metric, 1.0)
        return value * multiplier
    
    async def start(self) -> None:
        """Start the real-time monitor."""
        if self._running:
            logger.warning("Monitor is already running")
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Real-time monitor started")
    
    async def stop(self) -> None:
        """Stop the real-time monitor."""
        if not self._running:
            logger.warning("Monitor is not running")
            return
            
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        # Save current data
        self._save_metrics()
        logger.info("Real-time monitor stopped")
    
    async def _update_loop(self) -> None:
        """Background task to update metrics regularly."""
        while self._running:
            try:
                # Fetch new metrics from external sources
                await self._fetch_external_metrics()
                
                # Process metrics and check thresholds
                self._process_metrics()
                
                # Update adaptive thresholds
                self._update_thresholds()
                
                # Save metrics periodically
                self._save_metrics()
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
            
            # Wait for next update interval
            await asyncio.sleep(self.update_interval)
    
    async def _fetch_external_metrics(self) -> None:
        """Fetch metrics from external sources like social media APIs."""
        # This would normally call various platform APIs to get metrics
        # For now, we'll just simulate with dummy data
        
        # In a real implementation, this would be replaced with actual API calls
        # to Twitter, Instagram, etc.
        
        current_time = time.time()
        
        for platform in self.platforms:
            # Simulate fetching engagement metrics
            engagement = np.random.normal(loc=0.03, scale=0.01)  # ~3% engagement rate
            likes = np.random.poisson(5)  # ~5 likes per interval
            shares = np.random.poisson(2)  # ~2 shares per interval
            comments = np.random.poisson(3)  # ~3 comments per interval
            views = np.random.poisson(20)  # ~20 views per interval
            
            # Record metrics
            await self.record_metric(
                name="engagement_rate",
                value=engagement,
                platform=platform,
                timestamp=current_time
            )
            
            await self.record_metric(
                name="likes_per_minute",
                value=likes / (self.update_interval / 60),
                platform=platform,
                timestamp=current_time
            )
            
            await self.record_metric(
                name="shares_per_minute",
                value=shares / (self.update_interval / 60),
                platform=platform,
                timestamp=current_time
            )
            
            await self.record_metric(
                name="comments_per_minute",
                value=comments / (self.update_interval / 60),
                platform=platform,
                timestamp=current_time
            )
            
            await self.record_metric(
                name="views_per_minute",
                value=views / (self.update_interval / 60),
                platform=platform,
                timestamp=current_time
            )
    
    async def record_metric(
        self,
        name: str,
        value: float,
        platform: str = "all",
        content_id: str = "",
        timestamp: Optional[float] = None
    ) -> None:
        """
        Record a new performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            platform: Platform the metric is for
            content_id: Optional content identifier
            timestamp: Optional timestamp (defaults to current time)
        """
        if platform not in self.metrics:
            self.metrics[platform] = {}
            
        if name not in self.metrics[platform]:
            self.metrics[platform][name] = []
        
        # Create metric
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=timestamp or time.time(),
            platform=platform,
            content_id=content_id
        )
        
        # Store metric
        self.metrics[platform][name].append(metric)
        
        # Also store in "all" if this is platform-specific
        if platform != "all":
            if name not in self.metrics["all"]:
                self.metrics["all"][name] = []
            self.metrics["all"][name].append(metric)
        
        # Check thresholds immediately for real-time alerting
        await self._check_threshold(metric)
        
        # Integrate with neural enhancer if available
        if self.neural_enhancer:
            try:
                self.neural_enhancer.process_new_metric(metric.to_dict())
            except Exception as e:
                logger.error(f"Error integrating with neural enhancer: {e}")
    
    async def _check_threshold(self, metric: PerformanceMetric) -> None:
        """Check if metric exceeds threshold and generate alert if needed."""

