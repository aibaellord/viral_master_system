from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .base_platform_client import BasePlatformClient
from ..types.metrics import (
    EngagementMetrics,
    ViralMetrics,
    AudienceRetentionMetrics,
    VideoPerformanceMetrics,
)
from ..utils.rate_limiter import RateLimiter
from ..utils.exceptions import APIError, AuthenticationError

logger = logging.getLogger(__name__)

class YouTubeClient(BasePlatformClient):
    """YouTube platform client for interacting with YouTube Data and Analytics APIs."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
        rate_limit: int = 10000,
        rate_period: int = 86400,  # 24 hours in seconds
    ):
        """Initialize YouTube client with OAuth2 credentials and rate limiting.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 redirect URI
            scopes: List of required API scopes
            rate_limit: Number of requests allowed per rate_period
            rate_period: Time period for rate limiting in seconds
        """
        super().__init__()
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or [
            'https://www.googleapis.com/auth/youtube.readonly',
            'https://www.googleapis.com/auth/youtube.force-ssl',
            'https://www.googleapis.com/auth/youtubepartner',
            'https://www.googleapis.com/auth/yt-analytics.readonly',
        ]
        
        self.rate_limiter = RateLimiter(rate_limit, rate_period)
        self._youtube_service = None
        self._analytics_service = None

    async def authenticate(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        """Authenticate with YouTube API using OAuth2.
        
        Args:
            credentials: Optional dictionary containing OAuth2 credentials
        
        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            if credentials:
                creds = Credentials.from_authorized_user_info(credentials, self.scopes)
            else:
                # Implement OAuth2 flow for new authentication
                flow = await self._create_oauth_flow()
                creds = await self._complete_oauth_flow(flow)
            
            self._youtube_service = build('youtube', 'v3', credentials=creds)
            self._analytics_service = build('youtubeAnalytics', 'v2', credentials=creds)
        except Exception as e:
            raise AuthenticationError(f"Failed to authenticate with YouTube: {str(e)}")

    async def get_video_metrics(self, video_id: str) -> VideoPerformanceMetrics:
        """Get comprehensive metrics for a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            VideoPerformanceMetrics containing all relevant metrics
        """
        await self.rate_limiter.acquire()
        
        try:
            # Get basic video statistics
            video_response = await self._execute_request(
                lambda: self._youtube_service.videos().list(
                    part='statistics,contentDetails',
                    id=video_id
                ).execute()
            )
            
            if not video_response.get('items'):
                raise APIError(f"Video {video_id} not found")
                
            stats = video_response['items'][0]['statistics']
            
            # Get analytics data
            analytics_response = await self._execute_request(
                lambda: self._analytics_service.reports().query(
                    ids=f'channel==MINE',
                    metrics='estimatedMinutesWatched,averageViewDuration,averageViewPercentage',
                    filters=f'video=={video_id}',
                    dimensions='video',
                    startDate='2020-01-01',
                    endDate=datetime.now().strftime('%Y-%m-%d')
                ).execute()
            )
            
            return VideoPerformanceMetrics(
                views=int(stats.get('viewCount', 0)),
                likes=int(stats.get('likeCount', 0)),
                comments=int(stats.get('commentCount', 0)),
                shares=await self._get_share_count(video_id),
                watch_time=float(analytics_response.get('rows', [[0, 0, 0]])[0][0]),
                avg_view_duration=float(analytics_response.get('rows', [[0, 0, 0]])[0][1]),
                retention_rate=float(analytics_response.get('rows', [[0, 0, 0]])[0][2]),
                viral_coefficient=await self._calculate_viral_coefficient(video_id)
            )
        except HttpError as e:
            raise APIError(f"YouTube API error: {str(e)}")

    async def get_audience_retention(self, video_id: str) -> AudienceRetentionMetrics:
        """Get detailed audience retention metrics for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            AudienceRetentionMetrics containing retention data
        """
        await self.rate_limiter.acquire()
        
        try:
            retention_data = await self._execute_request(
                lambda: self._analytics_service.reports().query(
                    ids=f'channel==MINE',
                    metrics='audienceWatchRatio,relativeRetentionPerformance',
                    filters=f'video=={video_id}',
                    dimensions='elapsedVideoTimeRatio',
                    startDate='2020-01-01',
                    endDate=datetime.now().strftime('%Y-%m-%d')
                ).execute()
            )
            
            return AudienceRetentionMetrics(
                retention_points=retention_data.get('rows', []),
                average_view_percentage=float(retention_data.get('averageViewPercentage', 0)),
                relative_retention=float(retention_data.get('relativeRetentionPerformance', 0))
            )
            
        except HttpError as e:
            raise APIError(f"Failed to get audience retention data: {str(e)}")

    async def get_viral_metrics(self, video_id: str) -> ViralMetrics:
        """Calculate viral metrics for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            ViralMetrics containing viral performance data
        """
        await self.rate_limiter.acquire()
        
        try:
            # Get social sharing and engagement data
            sharing_data = await self._execute_request(
                lambda: self._analytics_service.reports().query(
                    ids=f'channel==MINE',
                    metrics='sharingService,shares',
                    filters=f'video=={video_id}',
                    dimensions='sharingService',
                    startDate='2020-01-01',
                    endDate=datetime.now().strftime('%Y-%m-%d')
                ).execute()
            )
            
            # Calculate viral metrics
            viral_coef = await self._calculate_viral_coefficient(video_id)
            growth_rate = await self._calculate_growth_rate(video_id)
            
            return ViralMetrics(
                viral_coefficient=viral_coef,
                growth_rate=growth_rate,
                share_velocity=await self._calculate_share_velocity(video_id),
                social_impact_score=await self._calculate_social_impact(video_id)
            )
            
        except HttpError as e:
            raise APIError(f"Failed to get viral metrics: {str(e)}")

    async def _execute_request(self, request_func: callable) -> Any:
        """Execute an API request with retries and error handling.
        
        Args:
            request_func: Callable that executes the API request
            
        Returns:
            API response data
        """
        retries = 3
        for attempt in range(retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(None, request_func)
            except HttpError as e:
                if attempt == retries - 1:
                    raise
                if e.resp.status in (429, 500, 503):  # Rate limit or server error
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    async def _calculate_viral_coefficient(self, video_id: str) -> float:
        """Calculate viral coefficient based on shares and new viewers.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Viral coefficient as a float
        """
        # Implementation details for viral coefficient calculation
        shares = await self._get_share_count(video_id)
        new_viewers = await self._get_new_viewers_count(video_id)
        return shares / new_viewers if new_viewers > 0 else 0.0

    async def _calculate_growth_rate(self, video_id: str) -> float:
        """Calculate the growth rate of video views over time.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Growth rate as a float
        """
        # Implementation details for growth rate calculation
        current_period = await self._get_period_views(video_id, days=7)
        previous_period = await self._get_period_views(video_id, days=14, offset=7)
        return (current_period - previous_period) / previous_period if previous_period > 0 else 0.0

    async def _calculate_share_velocity(self, video_id: str) -> float:
        """Calculate the velocity of shares over time.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Share velocity as a float
        """
        # Implementation details for share velocity calculation
        shares_timeline = await self._get_shares_timeline(video_id)
        return sum(shares_timeline) / len(shares_timeline) if shares_timeline else 0.0

    async def _calculate_social_impact(self, video_id: str) -> float:
        """Calculate social impact score based on engagement metrics.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Social impact score as a float
        """
        metrics = await self.get_video_metrics(video_id)
        weighted_score = (
            metrics.views * 1.0 +
            metrics.likes * 2.0 +
            metrics.comments * 3.0 +
            metrics.shares * 4.0
        ) / 10.0
        return min(weighted_score, 100.0)

