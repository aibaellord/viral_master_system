from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from typing import Any, Dict, List, Optional, Union
import logging
import json
from enum import Enum
import aiohttp
import backoff
from ratelimit import limits, RateLimitException

from ..metrics.config import PlatformConfig
from ..metrics.metrics_aggregator import MetricsSnapshot

logger = logging.getLogger(__name__)

class PlatformError(Exception):
    """Base exception for platform-related errors"""
    pass

class AuthenticationError(PlatformError):
    """Raised when authentication fails"""
    pass

class RateLimitError(PlatformError):
    """Raised when rate limits are exceeded"""
    pass

class MetricType(Enum):
    ENGAGEMENT = "engagement"
    REACH = "reach"
    VIEWS = "views"
    FOLLOWERS = "followers"
    LIKES = "likes"
    COMMENTS = "comments"
    SHARES = "shares"
    VIRAL_COEFFICIENT = "viral_coefficient"

@dataclass
class PlatformMetrics:
    """Common metrics across platforms"""
    platform_id: str
    timestamp: datetime
    engagement_rate: float
    reach: int
    views: int
    followers: int
    likes: int
    comments: int
    shares: int
    viral_coefficient: float
    raw_data: Dict[str, Any]

class BasePlatformClient(ABC):
    """Base class for platform-specific API clients"""
    
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_rate_limits()
        self._setup_logging()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _setup_rate_limits(self):
        """Configure rate limiting based on platform config"""
        self.calls_per_second = self.config.rate_limit.calls_per_second
        self.calls_per_minute = self.config.rate_limit.calls_per_minute
        
        # Decorate methods with rate limits
        self._make_request = limits(
            calls=self.calls_per_second,
            period=1
        )(self._make_request)
        
    def _setup_logging(self):
        """Configure logging for the platform client"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def authenticate(self) -> None:
        """Authenticate with the platform API"""
        pass
        
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that the current credentials are valid"""
        pass

    @backoff.on_exception(
        backoff.expo,
        (RateLimitException, aiohttp.ClientError),
        max_tries=5
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict:
        """Make an HTTP request with retries and error handling"""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        headers = headers or {}
        headers.update(self.get_auth_headers())
        
        try:
            async with self.session.request(
                method,
                f"{self.config.api_base_url}{endpoint}",
                params=params,
                json=data,
                headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                raise AuthenticationError("Authentication failed")
            elif e.status == 429:
                raise RateLimitError("Rate limit exceeded")
            raise PlatformError(f"Request failed: {str(e)}")
            
        except aiohttp.ClientError as e:
            raise PlatformError(f"Request error: {str(e)}")

    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        pass

    @abstractmethod
    async def get_metrics(self, timeframe: timedelta) -> List[PlatformMetrics]:
        """Get platform metrics for the specified timeframe"""
        pass

    @abstractmethod
    async def calculate_viral_coefficient(self, metrics: List[PlatformMetrics]) -> float:
        """Calculate viral coefficient from metrics"""
        pass

    async def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot"""
        metrics = await self.get_metrics(timedelta(hours=1))
        viral_coef = await self.calculate_viral_coefficient(metrics)
        
        if not metrics:
            raise PlatformError("Failed to retrieve metrics")
            
        latest = metrics[-1]
        return MetricsSnapshot(
            platform_id=self.config.platform_id,
            timestamp=datetime.utcnow(),
            engagement_rate=latest.engagement_rate,
            reach=latest.reach,
            views=latest.views, 
            followers=latest.followers,
            likes=latest.likes,
            comments=latest.comments,
            shares=latest.shares,
            viral_coefficient=viral_coef
        )

    async def monitor_metrics(
        self,
        interval: int = 60,
        callback: Optional[callable] = None
    ) -> None:
        """Monitor metrics continuously with optional callback"""
        while True:
            try:
                snapshot = await self.get_snapshot()
                if callback:
                    await callback(snapshot)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error monitoring metrics: {str(e)}")
                await asyncio.sleep(interval)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from typing import Any, Dict, List, Optional, Union
import aiohttp
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from ratelimit import limits, sleep_and_retry

@dataclass
class RateLimitConfig:
    calls: int
    period: int  # in seconds
    retry_after: int = 60  # default retry after 60 seconds

@dataclass
class MetricsConfig:
    collection_interval: int  # in seconds
    batch_size: int = 100

@dataclass
class ClientConfig:
    api_key: str
    base_url: str
    rate_limit: RateLimitConfig
    metrics_config: MetricsConfig
    timeout: int = 30
    max_retries: int = 3

class PlatformClientError(Exception):
    """Base exception for platform client errors."""
    pass

class RateLimitExceeded(PlatformClientError):
    """Raised when rate limit is exceeded."""
    pass

class AuthenticationError(PlatformClientError):
    """Raised when authentication fails."""
    pass

class APIError(PlatformClientError):
    """Raised when API returns an error."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class BasePlatformClient(ABC):
    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._metrics_queue: asyncio.Queue = asyncio.Queue()
        self._last_request_time: Dict[str, datetime] = {}
    
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def initialize(self):
        """Initialize the client session and start background tasks."""
        self._session = aiohttp.ClientSession(
            base_url=self.config.base_url,
            headers=self._get_default_headers()
        )
        asyncio.create_task(self._process_metrics_queue())

    async def cleanup(self):
        """Cleanup resources and close sessions."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"{self.__class__.__name__}/1.0"
        }

    @sleep_and_retry
    @limits(calls=lambda self: self.config.rate_limit.calls,
            period=lambda self: self.config.rate_limit.period)
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an API request with rate limiting and retries."""
        if not self._session:
            raise PlatformClientError("Client not initialized")

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=lambda e: isinstance(e, (APIError, aiohttp.ClientError))
        )
        async def _do_request():
            try:
                async with self._session.request(
                    method,
                    endpoint,
                    **kwargs,
                    timeout=self.config.timeout
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", self.config.rate_limit.retry_after))
                        raise RateLimitExceeded(f"Rate limit exceeded. Retry after {retry_after} seconds")
                    
                    if response.status == 401:
                        raise AuthenticationError("Authentication failed")
                    
                    if not 200 <= response.status < 300:
                        error_data = await response.json()
                        raise APIError(response.status, error_data.get("message", "Unknown error"))
                    
                    return await response.json()
            except asyncio.TimeoutError:
                raise PlatformClientError("Request timed out")

        return await _do_request()

    async def _collect_metrics(self, metric_name: str, values: Dict[str, Any]):
        """Queue metrics for collection."""
        await self._metrics_queue.put({
            "name": metric_name,
            "timestamp": datetime.utcnow().isoformat(),
            "values": values
        })

    async def _process_metrics_queue(self):
        """Process queued metrics in batches."""
        while True:
            metrics_batch = []
            try:
                while len(metrics_batch) < self.config.metrics_config.batch_size:
                    try:
                        metric = await asyncio.wait_for(
                            self._metrics_queue.get(),
                            timeout=self.config.metrics_config.collection_interval
                        )
                        metrics_batch.append(metric)
                    except asyncio.TimeoutError:
                        break
                
                if metrics_batch:
                    await self._send_metrics_batch(metrics_batch)
            
            except Exception as e:
                self.logger.error(f"Error processing metrics batch: {e}")
                
            await asyncio.sleep(1)

    @abstractmethod
    async def _send_metrics_batch(self, metrics: List[Dict[str, Any]]):
        """Send collected metrics to the platform."""
        raise NotImplementedError()

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the platform."""
        raise NotImplementedError()

    @abstractmethod
    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        raise NotImplementedError()

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the platform API is healthy."""
        raise NotImplementedError()

