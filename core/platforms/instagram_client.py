from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from ratelimit import limits, sleep_and_retry

from .base_platform_client import BasePlatformClient, PlatformMetrics, AuthenticationError, RateLimitError

@dataclass
class InstagramMetrics(PlatformMetrics):
    follower_count: int
    following_count: int
    post_count: int
    engagement_rate: float
    avg_likes: float
    avg_comments: float
    viral_coefficient: float
    story_views: Optional[float] = None
    reels_plays: Optional[float] = None
    profile_visits: Optional[int] = None
    website_clicks: Optional[int] = None
    reach: Optional[int] = None
    impressions: Optional[int] = None

class InstagramClient(BasePlatformClient):
    GRAPH_API_VERSION = "v18.0"
    BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
    
    # Instagram API rate limits (adjust based on your app tier)
    CALLS_PER_HOUR = 200
    CALLS_PER_MINUTE = 20
    
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        access_token: str,
        client_config: Dict[str, Any]
    ):
        super().__init__(platform_name="instagram", client_config=client_config)
        self.app_id = app_id
        self.app_secret = app_secret
        self._access_token = access_token
        self._token_expiry: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def authenticate(self) -> None:
        """Authenticate with Instagram Graph API and handle token refresh."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            if not self._is_token_valid():
                await self._refresh_access_token()

        except aiohttp.ClientError as e:
            raise AuthenticationError(f"Failed to authenticate with Instagram: {str(e)}")

    async def _refresh_access_token(self) -> None:
        """Refresh the Instagram Graph API access token."""
        try:
            url = f"{self.BASE_URL}/oauth/access_token"
            params = {
                "grant_type": "fb_exchange_token",
                "client_id": self.app_id,
                "client_secret": self.app_secret,
                "fb_exchange_token": self._access_token
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise AuthenticationError(f"Token refresh failed: {await response.text()}")
                
                data = await response.json()
                self._access_token = data["access_token"]
                self._token_expiry = datetime.now() + timedelta(seconds=data["expires_in"])
                
        except (aiohttp.ClientError, KeyError) as e:
            raise AuthenticationError(f"Failed to refresh access token: {str(e)}")

    def _is_token_valid(self) -> bool:
        """Check if the current access token is valid and not expired."""
        return bool(
            self._token_expiry 
            and self._token_expiry > datetime.now() + timedelta(minutes=5)
        )

    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a rate-limited request to the Instagram Graph API."""
        if not self._is_token_valid():
            await self.authenticate()

        params = params or {}
        params["access_token"] = self._access_token

        try:
            async with self._session.get(f"{self.BASE_URL}/{endpoint}", params=params) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds.")
                
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

    async def collect_metrics(self, profile_id: str) -> InstagramMetrics:
        """Collect comprehensive metrics for an Instagram profile."""
        try:
            # Fetch basic profile metrics
            profile_data = await self._make_request(f"{profile_id}", {
                "fields": "followers_count,follows_count,media_count"
            })

            # Fetch recent media metrics
            media_data = await self._make_request(f"{profile_id}/media", {
                "fields": "like_count,comments_count,engagement,impressions,reach",
                "limit": 50
            })

            # Calculate engagement metrics
            posts = media_data.get("data", [])
            total_likes = sum(post.get("like_count", 0) for post in posts)
            total_comments = sum(post.get("comments_count", 0) for post in posts)
            total_engagement = sum(post.get("engagement", 0) for post in posts)
            post_count = len(posts)

            if post_count > 0:
                avg_likes = total_likes / post_count
                avg_comments = total_comments / post_count
                engagement_rate = (total_engagement / post_count) / profile_data["followers_count"] * 100
            else:
                avg_likes = avg_comments = engagement_rate = 0.0

            # Calculate viral coefficient (simplified version)
            viral_coefficient = self._calculate_viral_coefficient(posts)

            return InstagramMetrics(
                follower_count=profile_data["followers_count"],
                following_count=profile_data["follows_count"],
                post_count=profile_data["media_count"],
                engagement_rate=engagement_rate,
                avg_likes=avg_likes,
                avg_comments=avg_comments,
                viral_coefficient=viral_coefficient,
                # Optional metrics that require additional API calls
                story_views=await self._get_story_metrics(profile_id),
                reels_plays=await self._get_reels_metrics(profile_id),
                profile_visits=await self._get_profile_visits(profile_id),
                website_clicks=await self._get_website_clicks(profile_id),
                reach=await self._get_account_reach(profile_id),
                impressions=await self._get_account_impressions(profile_id)
            )

        except Exception as e:
            self.logger.error(f"Failed to collect Instagram metrics: {str(e)}")
            raise

    def _calculate_viral_coefficient(self, posts: List[Dict]) -> float:
        """Calculate viral coefficient based on engagement and sharing patterns."""
        try:
            if not posts:
                return 0.0

            total_shares = sum(post.get("shares", 0) for post in posts)
            total_reach = sum(post.get("reach", 0) for post in posts)
            
            if total_reach == 0:
                return 0.0

            # Viral coefficient = (shares * avg_shared_post_reach) / original_reach
            return (total_shares * (total_reach / len(posts))) / total_reach

        except Exception as e:
            self.logger.warning(f"Failed to calculate viral coefficient: {str(e)}")
            return 0.0

    async def _get_story_metrics(self, profile_id: str) -> Optional[float]:
        """Fetch and calculate average story views."""
        try:
            stories = await self._make_request(f"{profile_id}/stories", {
                "fields": "insights.metric(impressions)"
            })
            if not stories.get("data"):
                return None

            total_views = sum(
                story.get("insights", {}).get("data", [{}])[0].get("values", [{}])[0].get("value", 0)
                for story in stories["data"]
            )
            return total_views / len(stories["data"])

        except Exception as e:
            self.logger.warning(f"Failed to fetch story metrics: {str(e)}")
            return None

    async def _get_reels_metrics(self, profile_id: str) -> Optional[float]:
        """Fetch and calculate average reels plays."""
        try:
            reels = await self._make_request(f"{profile_id}/reels", {
                "fields": "plays"
            })
            if not reels.get("data"):
                return None

            total_plays = sum(reel.get("plays", 0) for reel in reels["data"])
            return total_plays / len(reels["data"])

        except Exception as e:
            self.logger.warning(f"Failed to fetch reels metrics: {str(e)}")
            return None

    async def check_health(self) -> bool:
        """Check the health of the Instagram API connection."""
        try:
            await self.authenticate()
            # Make a light API call to verify connectivity
            await self._make_request("app", {"fields": "id"})
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    async def close(self) -> None:
        """Close the client session and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import aiohttp
from .base_platform_client import (
    BasePlatformClient,
    ClientConfig,
    RateLimitConfig,
    MetricsConfig,
    PlatformClientError,
    AuthenticationError,
    APIError
)

class InstagramClient(BasePlatformClient):
    """Instagram Graph API client implementation."""
    
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self._rate_limits: Dict[str, Any] = {}

    async def authenticate(self) -> bool:
        """Authenticate with Instagram Graph API using OAuth."""
        try:
            auth_response = await self._make_request(
                "GET",
                "/oauth/access_token",
                params={
                    "grant_type": "client_credentials",
                    "client_id": self.config.api_key,
                    "client_secret": self.config.api_secret,
                    "scope": "instagram_basic,instagram_manage_insights"
                }
            )
            
            self.access_token = auth_response["access_token"]
            self.token_expires_at = datetime.utcnow() + timedelta(seconds=auth_response["expires_in"])
            
            # Update session headers with new token
            if self._session:
                self._session.headers.update({"Authorization": f"Bearer {self.access_token}"})
            
            return True
        except APIError as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get Instagram API rate limit status."""
        try:
            response = await self._make_request(
                "GET",
                "/debug_token",
                params={"input_token": self.access_token}
            )
            
            self._rate_limits = {
                "calls_remaining": response["rate_limit_remaining"],
                "calls_made": response["rate_limit_usage"],
                "reset_time": datetime.fromtimestamp(response["rate_limit_reset"]).isoformat()
            }
            
            return self._rate_limits
        except APIError as e:
            self.logger.error(f"Failed to get rate limits: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check Instagram API health status."""
        try:
            # First check authentication status
            if not self.access_token or (
                self.token_expires_at and datetime.utcnow() >= self.token_expires_at
            ):
                await self.authenticate()

            # Make a light API call to check health
            response = await self._make_request(
                "GET",
                "/me",
                params={"fields": "id"}
            )
            
            return "id" in response
        except PlatformClientError:
            return False

    async def _send_metrics_batch(self, metrics: List[Dict[str, Any]]):
        """Send collected Instagram metrics to analytics backend."""
        try:
            # Enrich metrics with Instagram-specific metadata
            enriched_metrics = []
            for metric in metrics:
                enriched_metric = {
                    **metric,
                    "platform": "instagram",
                    "api_version": "v12.0",
                    "rate_limits": self._rate_limits
                }
                enriched_metrics.append(enriched_metric)

            # Send metrics to analytics endpoint
            await self._make_request(
                "POST",
                "/analytics/metrics",
                json={
                    "metrics": enriched_metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except APIError as e:
            self.logger.error(f"Failed to send metrics batch: {e}")

    async def get_user_insights(self, user_id: str, metrics: List[str], period: str = "day") -> Dict[str, Any]:
        """Get Instagram user insights."""
        try:
            response = await self._make_request(
                "GET",
                f"/{user_id}/insights",
                params={
                    "metric": ",".join(metrics),
                    "period": period
                }
            )

            # Queue metrics for collection
            await self._collect_metrics("user_insights", {
                "user_id": user_id,
                "metrics": response["data"],
                "period": period
            })

            return response["data"]
        except APIError as e:
            self.logger.error(f"Failed to get user insights: {e}")
            return {}

    async def get_media_insights(self, media_id: str) -> Dict[str, Any]:
        """Get insights for a specific media object."""
        try:
            response = await self._make_request(
                "GET",
                f"/{media_id}/insights",
                params={
                    "metric": "engagement,impressions,reach,saved"
                }
            )

            # Queue metrics for collection
            await self._collect_metrics("media_insights", {
                "media_id": media_id,
                "metrics": response["data"]
            })

            return response["data"]
        except APIError as e:
            self.logger.error(f"Failed to get media insights: {e}")
            return {}

    async def analyze_account_growth(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze account growth over time."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            response = await self._make_request(
                "GET",
                f"/{user_id}/insights",
                params={
                    "metric": "follower_count,profile_views,reach",
                    "period": "day",
                    "since": int(start_time.timestamp()),
                    "until": int(end_time.timestamp())
                }
            )

            growth_metrics = {
                "follower_growth": response["data"][0]["values"],
                "profile_engagement": response["data"][1]["values"],
                "reach_trends": response["data"][2]["values"]
            }

            # Queue analytics for collection
            await self._collect_metrics("account_growth", {
                "user_id": user_id,
                "period_days": days,
                "metrics": growth_metrics
            })

            return growth_metrics
        except APIError as e:
            self.logger.error(f"Failed to analyze account growth: {e}")
            return {}

