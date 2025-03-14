from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from uuid import UUID, uuid4
import networkx as nx
from prometheus_client import Counter, Gauge, Histogram

from .viral_orchestrator import ViralOrchestrator
from .strategy_optimizer import StrategyOptimizer
from .metrics_collector import MetricsCollector
from .config_manager import ConfigManager
from .monitoring_system import MonitoringSystem

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    id: UUID
    name: str
    priority: TaskPriority
    dependencies: Set[UUID]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retries: int
    max_retries: int
    payload: Dict[str, Any]
    result: Optional[Any]
    error: Optional[Exception]

class TaskAutomator:
    def __init__(
        self,
        orchestrator: ViralOrchestrator,
        optimizer: StrategyOptimizer,
        metrics: MetricsCollector,
        config: ConfigManager,
        monitor: MonitoringSystem
    ):
        self.orchestrator = orchestrator
        self.optimizer = optimizer
        self.metrics = metrics
        self.config = config
        self.monitor = monitor
        
        self.task_graph = nx.DiGraph()
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[UUID, Task] = {}
        
        # Metrics
        self.task_counter = Counter('tasks_total', 'Total number of tasks processed')
        self.active_tasks = Gauge('active_tasks', 'Number of currently running tasks')
        self.task_duration = Histogram('task_duration_seconds', 'Task execution duration')
        
        self.logger = self.config.get_logger(__name__)

    async def create_task(
        self,
        name: str,
        priority: TaskPriority,
        dependencies: Optional[Set[UUID]] = None,
        payload: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Task:
        """Create a new task with the specified parameters."""
        task = Task(
            id=uuid4(),
            name=name,
            priority=priority,
            dependencies=dependencies or set(),
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            retries=0,
            max_retries=max_retries,
            payload=payload or {},
            result=None,
            error=None
        )
        
        self.task_graph.add_node(task.id, task=task)
        for dep_id in task.dependencies:
            self.task_graph.add_edge(dep_id, task.id)
            
        await self.task_queue.put((priority.value, task.id))
        self.task_counter.inc()
        
        self.logger.info(f"Created task {task.id}: {name}")
        return task

    async def execute_task(self, task: Task) -> None:
        """Execute a single task with proper error handling and metrics collection."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self.running_tasks[task.id] = task
            self.active_tasks.inc()
            
            with self.task_duration.time():
                # Delegate task execution to appropriate component
                if task.name.startswith("optimize_"):
                    result = await self.optimizer.handle_task(task.payload)
                elif task.name.startswith("metric_"):
                    result = await self.metrics.process_task(task.payload)
                else:
                    result = await self.orchestrator.execute_task(task.payload)
                
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {str(e)}", exc_info=True)
            task.error = e
            
            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRYING
                task.retries += 1
                await self.task_queue.put((task.priority.value, task.id))
            else:
                task.status = TaskStatus.FAILED
                await self.handle_task_failure(task)
                
        finally:
            self.active_tasks.dec()
            del self.running_tasks[task.id]

    async def handle_task_failure(self, task: Task) -> None:
        """Handle task failures with recovery procedures."""
        self.logger.error(f"Task {task.id} failed permanently after {task.retries} retries")
        await self.monitor.report_incident(
            component="TaskAutomator",
            severity="high",
            message=f"Task {task.id} ({task.name}) failed permanently",
            details={"error": str(task.error), "retries": task.retries}
        )
        
        # Notify dependent tasks
        for successor in self.task_graph.successors(task.id):
            successor_task = self.task_graph.nodes[successor]["task"]
            successor_task.status = TaskStatus.FAILED
            successor_task.error = Exception(f"Dependency {task.id} failed")

    async def optimize_resources(self) -> None:
        """Optimize resource allocation based on current task load."""
        total_tasks = len(self.running_tasks)
        high_priority = sum(1 for t in self.running_tasks.values() if t.priority == TaskPriority.HIGH)
        
        resource_metrics = {
            "total_tasks": total_tasks,
            "high_priority_tasks": high_priority,
            "queue_size": self.task_queue.qsize()
        }
        
        await self.optimizer.optimize_resources(resource_metrics)
        await self.monitor.update_resource_status(resource_metrics)

    def get_task_dependencies(self, task_id: UUID) -> List[Task]:
        """Get all dependencies for a given task."""
        return [
            self.task_graph.nodes[dep_id]["task"]
            for dep_id in self.task_graph.predecessors(task_id)
        ]

    async def visualize_workflow(self) -> str:
        """Generate a visualization of the current task workflow."""
        dot_graph = nx.drawing.nx_pydot.to_pydot(self.task_graph)
        for node in dot_graph.get_nodes():
            task_id = UUID(node.get_name())
            if task_id in self.task_graph.nodes:
                task = self.task_graph.nodes[task_id]["task"]
                node.set_label(f"{task.name}\n{task.status.name}")
                node.set_color(self._get_status_color(task.status))
        
        return dot_graph.to_string()

    def _get_status_color(self, status: TaskStatus) -> str:
        """Get visualization color for task status."""
        colors = {
            TaskStatus.PENDING: "gray",
            TaskStatus.RUNNING: "blue",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.RETRYING: "orange"
        }
        return colors.get(status, "black")

    async def run(self) -> None:
        """Main loop for processing tasks."""
        self.logger.info("Starting TaskAutomator main loop")
        
        while True:
            try:
                # Resource optimization every 60 seconds
                if not self.task_queue.empty():
                    await self.optimize_resources()
                
                # Process next task
                priority, task_id = await self.task_queue.get()
                task = self.task_graph.nodes[task_id]["task"]
                
                # Check if dependencies are met
                dependencies = self.get_task_dependencies(task_id)
                if any(d.status != TaskStatus.COMPLETED for d in dependencies):
                    await self.task_queue.put((priority, task_id))
                    continue
                
                await self.execute_task(task)
                
            except Exception as e:
                self.logger.error("Error in TaskAutomator main loop", exc_info=True)
                await self.monitor.report_incident(
                    component="TaskAutomator",
                    severity="critical",
                    message="Main loop error",
                    details={"error": str(e)}
                )
                await asyncio.sleep(5)  # Backoff on error
                
            finally:
                self.task_queue.task_done()

