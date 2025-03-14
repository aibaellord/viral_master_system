#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workflow Manager for Content Optimization

This module implements a workflow manager that handles content optimization
workflows, manages state transitions, supports parallel processing, and
provides progress tracking and error handling.

It works seamlessly with the SystemOrchestrator and other components of the 
Viral Master System.
"""

import asyncio
import enum
import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic

import aiohttp
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """Represents possible states of a workflow."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Priority levels for workflow tasks."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class ContentType(str, Enum):
    """Types of content that can be processed."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTI = "multi"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    id: str
    name: str
    description: str
    handler: Callable
    required_resources: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300  # Timeout in seconds
    state: WorkflowState = WorkflowState.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Context information passed between workflow steps."""
    workflow_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    runtime_properties: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update workflow metrics."""
        self.metrics.update(metrics)
        self.updated_at = datetime.now()
    
    def add_step_result(self, step_id: str, result: Any) -> None:
        """Add a step result to the context."""
        self.step_results[step_id] = result
        self.updated_at = datetime.now()
    
    def get_step_result(self, step_id: str, default=None) -> Any:
        """Get a step result from the context."""
        return self.step_results.get(step_id, default)


class WorkflowProgressTracker:
    """Tracks and reports workflow progress."""
    
    def __init__(self, total_steps: int, workflow_id: str, description: str = ""):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.workflow_id = workflow_id
        self.description = description
        self.start_time = time.time()
        self.step_times: Dict[str, float] = {}
        self.current_step: Optional[str] = None
        self._pbar = tqdm(total=total_steps, desc=f"Workflow {workflow_id} {description}")
    
    def start_step(self, step_id: str, step_name: str) -> None:
        """Mark the start of a workflow step."""
        logger.info(f"Starting step {step_name} ({step_id}) for workflow {self.workflow_id}")
        self.current_step = step_id
        self.step_times[step_id] = time.time()
    
    def complete_step(self, step_id: str, step_name: str) -> None:
        """Mark a workflow step as completed."""
        self.completed_steps += 1
        duration = time.time() - self.step_times.get(step_id, self.start_time)
        logger.info(f"Completed step {step_name} ({step_id}) for workflow {self.workflow_id} in {duration:.2f}s")
        self._pbar.update(1)
        
    def fail_step(self, step_id: str, step_name: str, error: Optional[Exception] = None) -> None:
        """Mark a workflow step as failed."""
        duration = time.time() - self.step_times.get(step_id, self.start_time)
        error_msg = str(error) if error else "Unknown error"
        logger.error(f"Failed step {step_name} ({step_id}) for workflow {self.workflow_id} in {duration:.2f}s: {error_msg}")
        
    def get_progress(self) -> float:
        """Get the current progress as a percentage."""
        return (self.completed_steps / self.total_steps) * 100 if self.total_steps > 0 else 0
    
    def get_estimated_time_remaining(self) -> float:
        """Estimate the remaining time based on current progress."""
        if self.completed_steps == 0:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        steps_left = self.total_steps - self.completed_steps
        return (elapsed / self.completed_steps) * steps_left
    
    def close(self) -> None:
        """Close the progress tracker."""
        self._pbar.close()
        total_time = time.time() - self.start_time
        logger.info(f"Workflow {self.workflow_id} completed in {total_time:.2f}s")


class WorkflowError(Exception):
    """Base class for workflow-related exceptions."""
    pass


class StepExecutionError(WorkflowError):
    """Exception raised when a workflow step fails to execute."""
    def __init__(self, step_id: str, original_error: Exception):
        self.step_id = step_id
        self.original_error = original_error
        super().__init__(f"Error executing step {step_id}: {original_error}")


class WorkflowTimeoutError(WorkflowError):
    """Exception raised when a workflow or step times out."""
    pass


class WorkflowDefinitionError(WorkflowError):
    """Exception raised when there's an error in the workflow definition."""
    pass


class RetryPolicy:
    """Defines retry behavior for workflow steps."""
    
    def __init__(
        self, 
        max_retries: int = 3, 
        retry_interval: float = 5.0,
        backoff_factor: float = 2.0,
        retry_on_exceptions: List[Exception] = None,
        max_retry_interval: float = 60.0
    ):
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.backoff_factor = backoff_factor
        self.retry_on_exceptions = retry_on_exceptions or [Exception]
        self.max_retry_interval = max_retry_interval
    
    def should_retry(self, exception: Exception, retry_count: int) -> bool:
        """Determine if a retry should be attempted."""
        if retry_count >= self.max_retries:
            return False
        
        return any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions)
    
    def get_retry_interval(self, retry_count: int) -> float:
        """Calculate the interval before the next retry."""
        interval = self.retry_interval * (self.backoff_factor ** retry_count)
        return min(interval, self.max_retry_interval)


T = TypeVar('T')


class Workflow(Generic[T]):
    """
    Defines and manages a content optimization workflow.
    
    This class is responsible for:
    1. Defining the workflow structure
    2. Managing state transitions
    3. Handling error cases and retries
    4. Tracking progress
    5. Supporting both sync and async execution
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = "", 
        timeout: int = 3600,
        max_concurrency: int = 5,
        retry_policy: Optional[RetryPolicy] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[List[str]] = []  # List of steps that can run in parallel
        self.state = WorkflowState.PENDING
        self.context = WorkflowContext(workflow_id=self.id)
        self.progress_tracker: Optional[WorkflowProgressTracker] = None
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.retry_policy = retry_policy or RetryPolicy()
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[T] = None
        self.error: Optional[Exception] = None
        
    def add_step(
        self, 
        step_id: str, 
        name: str, 
        handler: Callable, 
        depends_on: List[str] = None,
        description: str = "",
        retry_policy: Optional[RetryPolicy] = None,
        timeout: int = 300,
        required_resources: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> 'Workflow':
        """
        Add a step to the workflow.
        
        Args:
            step_id: Unique identifier for the step
            name: Human-readable name for the step
            handler: Function that implements the step
            depends_on: List of step IDs that must complete before this step
            description: Detailed description of what the step does
            retry_policy: Custom retry policy for this step
            timeout: Maximum time (in seconds) the step can run
            required_resources: List of resources required for this step
            metadata: Additional information about the step
            
        Returns:
            The workflow instance (for method chaining)
        """
        depends_on = depends_on or []
        required_resources = required_resources or []
        metadata = metadata or {}
        
        # Validate step_id is unique
        if step_id in self.steps:
            raise WorkflowDefinitionError(f"Step with ID '{step_id}' already exists in the workflow")
        
        # Validate dependencies exist
        for dep_id in depends_on:
            if dep_id not in self.steps:
                raise WorkflowDefinitionError(f"Dependency '{dep_id}' not found in workflow")
        
        retry_policy_dict = {}
        if retry_policy:
            retry_policy_dict = {
                "max_retries": retry_policy.max_retries,
                "retry_interval": retry_policy.retry_interval,
                "backoff_factor": retry_policy.backoff_factor,
                "max_retry_interval": retry_policy.max_retry_interval
            }
        
        step = WorkflowStep(
            id=step_id,
            name=name,
            description=description,
            handler=handler,
            depends_on=depends_on,
            retry_policy=retry_policy_dict,
            timeout=timeout,
            required_resources=required_resources,
            metadata=metadata
        )
        
        self.steps[step_id] = step
        return self
    
    def _build_execution_order(self) -> None:
        """
        Build the execution order of steps based on dependencies.
        
        This method organizes steps into levels that can be executed in parallel.
        """
        remaining_steps = set(self.steps.keys())
        execution_levels = []
        
        while remaining_steps:
            # Find steps with all dependencies satisfied
            runnable = {
                step_id for step_id in remaining_steps
                if all(dep not in remaining_steps for dep in self.steps[step_id].depends_on)
            }
            
            if not runnable:
                # Circular dependency detected
                raise WorkflowDefinitionError(
                    f"Circular dependency detected in workflow {self.name}. "
                    f"Remaining steps: {remaining_steps}"
                )
            
            execution_levels.append(list(runnable))
            remaining_steps -= runnable
        
        self.execution_order = execution_levels
        logger.info(f"Built execution order for workflow {self.id}: {self.execution_order}")
    
    def validate(self) -> None:
        """
        Validate the workflow definition.
        
        Checks for:
        - Duplicate step IDs
        - Missing dependencies
        - Circular dependencies
        - Empty workflows
        """
        if not self.steps:
            raise WorkflowDefinitionError(f"Workflow '{self.name}' has no steps defined")
        
        # Check dependencies
        all_step_ids = set(self.steps.keys())
        for step_id, step in self.steps.items():
            for dep_id in step.depends_on:
                if dep_id not in all_step_ids:
                    raise WorkflowDefinitionError(
                        f"Step '{step_id}' depends on non-existent step '{dep_id}'"
                    )
        
        # Build execution order (also checks for circular dependencies)
        try:
            self._build_execution_order()
        except WorkflowDefinitionError as e:
            raise e
    
    async def _execute_step_async(self, step: WorkflowStep, context: WorkflowContext) -> Any:
        """Execute a single workflow step asynchronously."""
        step.state = WorkflowState.RUNNING
        step.start_time = time.time()
        
        if self.progress_tracker:
            self.progress_tracker.start_step(step.id, step.name)
        
        try:
            # Check if handler is async
            if asyncio.iscoroutinefunction(step.handler):
                result = await step.handler(context)
            else:
                # Run synchronous handlers in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: step.handler

