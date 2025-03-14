import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import aiohttp
from ratelimit import limits, sleep_and_retry

from .base_platform_client import BasePlatformClient, PlatformMetrics, ApiError, RateLimitError

@dataclass
class TikTokMetrics(PlatformMetrics):
    views: int
    likes: int
    shares: int
    comments: int
    watch_time: float  # in seconds
    completion_rate: float  # percentage
    engagement_rate: float
    follower_growth_rate: float
    viral_coefficient: float
    sound_usage: int
    duet_count: int
    stitch_count: int

class TikTokClient(BasePlatformClient):
    """TikTok API client implementing OAuth2 and comprehensive metrics collection."""
    
    API_BASE_URL = "https://open.tiktokapis.com/v2"
    RATE_LIMIT_CALLS = 100  # Requests per time period
    RATE_LIMIT_PERIOD = 300  # Time period in seconds (5 minutes)

    def __init__(
        self, 
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None
    ):
        super().__init__()
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expires_at: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Set up async context manager."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context manager."""
        if self._session:
            await self._session.close()

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make an API request with rate limiting and error handling."""
        if not self._session:
            raise RuntimeError("Client session not initialized. Use async context manager.")

        if not self._access_token or self._token_expired():
            await self._refresh_access_token()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self._session.request(
                method,
                f"{self.API_BASE_URL}/{endpoint.lstrip('/')}",
                headers=headers,
                **kwargs
            ) as response:
                if response.status == 429:
                    raise RateLimitError("TikTok API rate limit exceeded")
                elif response.status >= 400:
                    error_data = await response.json()
                    raise ApiError(f"TikTok API error: {error_data.get('error', 'Unknown error')}")
                
                return await response.json()
        except aiohttp.ClientError as e:
            raise ApiError(f"Request failed: {str(e)}")

    async def _refresh_access_token(self) -> None:
        """Refresh the OAuth2 access token."""
        if not self._refresh_token:
            raise ApiError("No refresh token available")

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.API_BASE_URL}/oauth/token",
                json=data
            ) as response:
                if response.status != 200:
                    raise ApiError("Failed to refresh access token")
                
                token_data = await response.json()
                self._access_token = token_data["access_token"]
                self._refresh_token = token_data["refresh_token"]
                self._token_expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])

    def _token_expired(self) -> bool:
        """Check if the access token has expired."""
        if not self._token_expires_at:
            return True
        return datetime.now() >= self._token_expires_at

    async def get_auth_url(self) -> str:
        """Get the OAuth2 authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "user.info.basic,video.list,video.stats"
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.API_BASE_URL}/oauth/authorize?{query}"

    async def complete_oauth(self, code: str) -> None:
        """Complete the OAuth2 flow with the received code."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.API_BASE_URL}/oauth/token",
                json=data
            ) as response:
                if response.status != 200:
                    raise ApiError("Failed to obtain access token")
                
                token_data = await response.json()
                self._access_token = token_data["access_token"]
                self._refresh_token = token_data["refresh_token"]
                self._token_expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])

    async def get_metrics(self, video_id: str) -> TikTokMetrics:
        """Get comprehensive metrics for a specific video."""
        response = await self._make_request("GET", f"/videos/{video_id}/stats")
        stats = response["stats"]
        
        # Calculate derived metrics
        total_engagements = stats["like_count"] + stats["comment_count"] + stats["share_count"]
        engagement_rate = total_engagements / stats["view_count"] if stats["view_count"] > 0 else 0
        
        # Calculate viral coefficient (shares per view)
        viral_coefficient = stats["share_count"] / stats["view_count"] if stats["view_count"] > 0 else 0

        return TikTokMetrics(
            views=stats["view_count"],
            likes=stats["like_count"],
            shares=stats["share_count"],
            comments=stats["comment_count"],
            watch_time=stats["total_watch_time"],
            completion_rate=stats["video_completion_rate"],
            engagement_rate=engagement_rate,
            follower_growth_rate=stats["follower_growth_rate"],
            viral_coefficient=viral_coefficient,
            sound_usage=stats["sound_usage_count"],
            duet_count=stats["duet_count"],
            stitch_count=stats["stitch_count"]
        )

    async def get_account_metrics(self) -> Dict:
        """Get metrics for the authenticated account."""
        response = await self._make_request("GET", "/account/stats")
        return response["stats"]

    async def get_video_analytics(
        self,
        video_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Get detailed analytics for a specific video over time."""
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        response = await self._make_request(
            "GET",
            f"/videos/{video_id}/analytics",
            params=params
        )
        return response["analytics"]

    async def check_rate_limits(self) -> Dict:
        """Check current rate limit status."""
        response = await self._make_request("GET", "/rate_limit_info")
        return response["rate_limit"]

    async def get_health_status(self) -> bool:
        """Check if the TikTok API is healthy and accessible."""
        try:
            await self._make_request("GET", "/health")
            return True
        except ApiError:
            return False

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import aiohttp
import jwt
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_platform_client import BasePlatformClient
from ..metrics import ViralMetrics, EngagementRate
from ..errors import RateLimitError, AuthenticationError, APIError

@dataclass
class TikTokMetrics:
    views: int
    likes: int
    comments: int
    shares: int
    watch_time: float
    completion_rate: float
    viral_coefficient: float
    growth_rate: float
    engagement_rate: float

class TikTokClient(BasePlatformClient):
    """TikTok API client for viral marketing system."""
    
    API_BASE_URL = "https://open.tiktokapis.com/v2"
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_PERIOD = 300  # 5 minutes in seconds

    def __init__(
        self,
        client_key: str,
        client_secret: str,
        redirect_uri: str,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None
    ):
        """Initialize TikTok client with OAuth2 credentials."""
        super().__init__()
        self.client_key = client_key
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expires_at: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = self.RATE_LIMIT_CALLS
        self._rate_limit_reset = datetime.now() + timedelta(seconds=self.RATE_LIMIT_PERIOD)

    async def __aenter__(self):
        """Set up async context manager."""
        self._session = aiohttp.ClientSession(base_url=self.API_BASE_URL)
        if self._access_token and self._refresh_token:
            await self._check_token_validity()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context manager."""
        if self._session:
            await self._session.close()
            self._session = None

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None
    ) -> Dict:
        """Make rate-limited API request to TikTok."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        if self._rate_limit_remaining <= 0:
            wait_time = (self._rate_limit_reset - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }

        async with self._session.request(
            method, endpoint, params=params, json=json, headers=headers
        ) as response:
            if response.status == 429:
                raise RateLimitError("TikTok API rate limit exceeded")
            elif response.status == 401:
                await self._refresh_access_token()
                return await self._make_request(method, endpoint, params, json)
            elif not response.ok:
                raise APIError(f"TikTok API error: {response.status}")

            self._update_rate_limits(response)
            return await response.json()

    async def _refresh_access_token(self):
        """Refresh OAuth2 access token."""
        if not self._refresh_token:
            raise AuthenticationError("No refresh token available")

        data = {
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.API_BASE_URL}/oauth/token",
                json=data
            ) as response:
                if not response.ok:
                    raise AuthenticationError("Failed to refresh access token")
                
                token_data = await response.json()
                self._access_token = token_data["access_token"]
                self._refresh_token = token_data["refresh_token"]
                self._token_expires_at = datetime.now() + timedelta(
                    seconds=token_data["expires_in"]
                )

    async def _check_token_validity(self):
        """Check if access token needs refresh."""
        if not self._token_expires_at:
            return

        if datetime.now() >= self._token_expires_at - timedelta(minutes=5):
            await self._refresh_access_token()

    def _update_rate_limits(self, response: aiohttp.ClientResponse):
        """Update rate limit tracking from response headers."""
        self._rate_limit_remaining = int(
            response.headers.get("X-RateLimit-Remaining", self.RATE_LIMIT_CALLS)
        )
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        self._rate_limit_reset = datetime.fromtimestamp(reset_time)

    async def get_video_metrics(self, video_id: str) -> TikTokMetrics:
        """Get comprehensive metrics for a specific video."""
        response = await self._make_request(
            "GET",
            f"/video/metrics",
            params={"video_id": video_id}
        )

        metrics = response["data"]["metrics"]
        return TikTokMetrics(
            views=metrics["play_count"],
            likes=metrics["like_count"],
            comments=metrics["comment_count"],
            shares=metrics["share_count"],
            watch_time=metrics["average_watch_time"],
            completion_rate=metrics["video_completion_rate"],
            viral_coefficient=self._calculate_viral_coefficient(metrics),
            growth_rate=self._calculate_growth_rate(metrics),
            engagement_rate=self._calculate_engagement_rate(metrics)
        )

    def _calculate_viral_coefficient(self, metrics: Dict[str, Any]) -> float:
        """Calculate viral coefficient based on shares and new viewer acquisition."""
        shares = metrics["share_count"]
        new_viewers = metrics["new_viewer_count"]
        return (new_viewers / shares) if shares > 0 else 0.0

    def _calculate_growth_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate growth rate over time periods."""
        current_period = metrics["current_period_views"]
        previous_period = metrics["previous_period_views"]
        return ((current_period - previous_period) / previous_period * 100) \
            if previous_period > 0 else 0.0

    def _calculate_engagement_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate engagement rate based on interactions."""
        total_interactions = (
            metrics["like_count"] +
            metrics["comment_count"] +
            metrics["share_count"]
        )
        return (total_interactions / metrics["play_count"] * 100) \
            if metrics["play_count"] > 0 else 0.0

    async def analyze_account_growth(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze account growth metrics over time period."""
        response = await self._make_request(
            "GET",
            "/account/metrics",
            params={
                "account_id": account_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        )

        daily_metrics = response["data"]["metrics"]
        return {
            "follower_growth": self._analyze_follower_growth(daily_metrics),
            "engagement_trends": self._analyze_engagement_trends(daily_metrics),
            "content_performance": self._analyze_content_performance(daily_metrics),
            "viral_impact": self._analyze_viral_impact(daily_metrics)
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def upload_video(
        self,
        video_path: str,
        description: str,
        **kwargs
    ) -> str:
        """Upload video to TikTok with retry logic."""
        # Init upload
        init_response = await self._make_request(
            "POST",
            "/video/upload/init",
            json={"title": description}
        )

        upload_url = init_response["data"]["upload_url"]
        video_id = init_response["data"]["video_id"]

        # Upload video chunks
        async with aiohttp.ClientSession() as session:
            with open(video_path, "rb") as video_file:
                await session.put(
                    upload_url,
                    data=video_file,
                    headers={"Content-Type": "video/mp4"}
                )

        # Publish video
        await self._make_request(
            "POST",
            "/video/publish",
            json={
                "video_id": video_id,
                "description": description,
                **kwargs
            }
        )

        return video_id

    async def get_oauth_url(self) -> str:
        """Get OAuth2 authorization URL."""
        params = {
            "client_key": self.client_key,
            "response_type": "code",
            "scope": "user.info.basic,video.list,video.upload",
            "redirect_uri": self.redirect_uri,
            "state": jwt.encode(
                {"timestamp": datetime.now().timestamp()},
                self.client_secret,
                algorithm="HS256"
            )
        }
        
        return (
            "https://www.tiktok.com/auth/authorize?" +
            "&".join(f"{k}={v}" for k, v in params.items())
        )

    async def handle_oauth_callback(self, code: str) -> None:
        """Handle OAuth2 callback and set access token."""
        data = {
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.API_BASE_URL}/oauth/token",
                json=data
            ) as response:
                if not response.ok:
                    raise AuthenticationError("Failed to obtain access token")
                
                token_data = await response.json()
                self._access_token = token_data["access_token"]
                self._refresh_token = token_data["refresh_token"]
                self._token_expires_at = datetime.now() + timedelta(
                    seconds=token_data["expires_in"]
                )

