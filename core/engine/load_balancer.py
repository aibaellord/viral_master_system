from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import time
from datetime import datetime
import numpy as np
from collections import defaultdict

class BalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    GEOGRAPHIC = "geographic"
    PREDICTIVE = "predictive"

@dataclass
class NodeHealth:
    status: bool
    last_check: datetime
    response_time: float
    error_rate: float
    resource_usage: Dict[str, float]

class LoadBalancer:
    def __init__(self, monitoring_system=None, task_automator=None):
        self.nodes: Dict[str, Dict] = defaultdict(dict)
        self.health_checks: Dict[str, NodeHealth] = {}
        self.monitoring_system = monitoring_system
        self.task_automator = task_automator
        self.current_strategy = BalancingStrategy.ROUND_ROBIN
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = defaultdict(lambda: {
            'requests': 0,
            'response_times': [],
            'errors': 0,
            'load': 0.0
        })
        
        # Predictive scaling parameters
        self.scaling_history = []
        self.scaling_thresholds = {
            'cpu_high': 0.75,
            'cpu_low': 0.25,
            'memory_high': 0.80,
            'memory_low': 0.30
        }

    async def register_node(self, node_id: str, capacity: float, region: str = None):
        """Register a new node with the load balancer."""
        self.nodes[node_id] = {
            'capacity': capacity,
            'current_load': 0.0,
            'region': region,
            'last_used': datetime.now(),
            'connections': 0
        }
        await self._initialize_health_check(node_id)

    async def distribute_load(self, request: Dict) -> str:
        """Distribute incoming requests based on current strategy."""
        strategy_map = {
            BalancingStrategy.ROUND_ROBIN: self._round_robin,
            BalancingStrategy.LEAST_CONNECTIONS: self._least_connections,
            BalancingStrategy.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin,
            BalancingStrategy.RESPONSE_TIME: self._response_time_based,
            BalancingStrategy.GEOGRAPHIC: self._geographic_routing,
            BalancingStrategy.PREDICTIVE: self._predictive_routing
        }
        
        return await strategy_map[self.current_strategy](request)

    async def _predictive_scaling(self):
        """Implement predictive scaling based on historical patterns."""
        current_metrics = await self.monitoring_system.get_system_metrics()
        prediction = self._calculate_resource_prediction(current_metrics)
        
        if prediction > self.scaling_thresholds['cpu_high']:
            await self._scale_up()
        elif prediction < self.scaling_thresholds['cpu_low']:
            await self._scale_down()

    async def _health_check_loop(self):
        """Continuous health monitoring of registered nodes."""
        while True:
            for node_id in self.nodes:
                health = await self._check_node_health(node_id)
                self.health_checks[node_id] = health
                
                if not health.status:
                    await self._handle_node_failure(node_id)
            
            await asyncio.sleep(30)  # Health check interval

    async def _optimize_resources(self):
        """Optimize resource allocation based on current usage patterns."""
        usage_patterns = await self.monitoring_system.get_usage_patterns()
        optimized_allocation = self._calculate_optimal_distribution(usage_patterns)
        
        for node_id, allocation in optimized_allocation.items():
            await self._adjust_node_resources(node_id, allocation)

    def _calculate_resource_prediction(self, current_metrics: Dict) -> float:
        """Calculate predicted resource needs using time series analysis."""
        historical_data = np.array(self.scaling_history[-24:])  # Last 24 data points
        weights = np.exp(np.linspace(-1, 0, len(historical_data)))
        weighted_average = np.average(historical_data, weights=weights)
        
        trend = np.polyfit(range(len(historical_data)), historical_data, 1)[0]
        prediction = weighted_average + trend
        
        return float(prediction)

    async def _handle_failover(self, failed_node: str):
        """Handle node failures and redistribute load."""
        self.logger.warning(f"Initiating failover for node {failed_node}")
        affected_tasks = await self.task_automator.get_node_tasks(failed_node)
        
        for task in affected_tasks:
            new_node = await self.distribute_load(task)
            await self.task_automator.reassign_task(task, new_node)

    async def update_strategy(self, strategy: BalancingStrategy):
        """Update the load balancing strategy."""
        self.current_strategy = strategy
        self.logger.info(f"Updated load balancing strategy to {strategy.value}")
        await self._optimize_resources()

    def get_performance_metrics(self) -> Dict:
        """Return current performance metrics."""
        return {
            'node_health': self.health_checks,
            'load_distribution': {node: data['current_load'] 
                                for node, data in self.nodes.items()},
            'response_times': self.metrics,
            'scaling_history': self.scaling_history
        }

