# Platform Integration Guide

This guide provides comprehensive documentation for integrating with various social media platforms, including authentication, metrics collection, error handling, and optimization guidelines.

## Table of Contents
- [TikTok Integration](#tiktok-integration)
- [YouTube Integration](#youtube-integration)
- [Implementing New Platforms](#implementing-new-platforms)
- [Error Handling](#error-handling)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Authentication](#authentication)
- [Optimization Guidelines](#optimization-guidelines)

## TikTok Integration

### Authentication
```python
from core.platforms import TikTokClient

client = TikTokClient(
    client_key="your_client_key",
    client_secret="your_client_secret",
    redirect_uri="your_redirect_uri"
)
```

### Available Metrics
- `view_count`: Total video views
- `share_count`: Number of shares
- `like_count`: Number of likes
- `comment_count`: Number of comments
- `follower_count`: Total followers
- `engagement_rate`: (likes + comments + shares) / views
- `video_retention`: Average watch time percentage
- `viral_coefficient`: Viral spread rate
- `audience_demographics`: Viewer demographics data
- `sound_usage`: Music/sound reuse statistics
- `hashtag_performance`: Hashtag engagement metrics

### Example: Collecting Metrics
```python
metrics = await client.collect_metrics(
    video_id="1234567890",
    metrics=[
        "view_count",
        "share_count",
        "engagement_rate",
        "viral_coefficient"
    ]
)
```

### Rate Limiting
```python
# Configure rate limits
client.configure_rate_limits(
    max_requests_per_second=10,
    max_requests_per_minute=300,
    max_requests_per_hour=3600
)
```

### Error Handling
```python
try:
    metrics = await client.collect_metrics(video_id="1234567890")
except TikTokRateLimitError:
    await asyncio.sleep(60)  # Wait for rate limit reset
except TikTokAuthError:
    await client.refresh_auth_token()
except TikTokAPIError as e:
    logger.error(f"TikTok API error: {e}")
```

## YouTube Integration

### Authentication
```python
from core.platforms import YouTubeClient

client = YouTubeClient(
    client_id="your_client_id",
    client_secret="your_client_secret",
    api_key="your_api_key"
)
```

### Available Metrics
- `view_count`: Total video views
- `watch_time`: Total watch time in minutes
- `average_view_duration`: Average view duration
- `engagement_rate`: (likes + comments) / views
- `subscriber_gain`: Subscribers gained from video
- `audience_retention`: Audience retention graph data
- `click_through_rate`: Click-through rate for cards/end screens
- `revenue`: Revenue metrics (if monetized)
- `playlist_adds`: Number of playlist additions
- `shares`: Share count
- `demographics`: Viewer demographics

### Example: Collecting Analytics
```python
analytics = await client.get_analytics(
    video_id="your_video_id",
    metrics=[
        "views",
        "watch_time",
        "average_view_duration",
        "engagement_rate"
    ],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Quota Management
```python
# Configure quota limits
client.configure_quota(
    daily_quota=10000,
    quota_costs={
        "analytics_read": 50,
        "data_read": 1,
        "video_upload": 1600
    }
)
```

### Error Handling
```python
try:
    analytics = await client.get_analytics(video_id="your_video_id")
except YouTubeQuotaExceeded:
    logger.warning("Daily quota exceeded")
    notify_quota_exceeded()
except YouTubeAuthError:
    await client.refresh_credentials()
except YouTubeAPIError as e:
    logger.error(f"YouTube API error: {e}")
```

## Implementing New Platforms

### Base Client Implementation
```python
from core.platforms import BasePlatformClient

class NewPlatformClient(BasePlatformClient):
    async def authenticate(self) -> None:
        # Implement platform-specific authentication
        pass

    async def collect_metrics(self, **kwargs) -> Dict[str, Any]:
        # Implement metric collection
        pass

    async def handle_rate_limits(self) -> None:
        # Implement rate limit handling
        pass
```

### Required Methods
- `authenticate()`: Handle OAuth or API key authentication
- `collect_metrics()`: Collect platform-specific metrics
- `handle_rate_limits()`: Implement rate limiting logic
- `validate_response()`: Validate API responses
- `transform_metrics()`: Transform platform metrics to standard format

## Error Handling

### Common Error Types
- `PlatformAuthError`: Authentication failures
- `PlatformRateLimitError`: Rate limit exceeded
- `PlatformAPIError`: General API errors
- `PlatformQuotaError`: API quota exceeded
- `PlatformDataError`: Invalid data responses

### Best Practices
1. Implement exponential backoff for rate limits
2. Cache authentication tokens appropriately
3. Validate all API responses
4. Log errors with context for debugging
5. Handle network timeouts gracefully

## Monitoring and Debugging

### Logging Configuration
```python
import logging

logging.config.dictConfig({
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'platform.log',
            'formatter': 'detailed'
        }
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'loggers': {
        'core.platforms': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG'
        }
    }
})
```

### Debugging Tools
1. Platform-specific API explorers
2. Request/response logging
3. Rate limit monitoring
4. Quota usage tracking
5. Performance profiling

## Authentication

### OAuth2 Flow
1. Register application with platform
2. Implement authorization code flow
3. Handle token refresh
4. Secure token storage
5. Monitor token expiration

### Best Practices
1. Use environment variables for credentials
2. Implement token refresh handling
3. Use secure storage for tokens
4. Monitor authentication failures
5. Implement retry logic

## Optimization Guidelines

### TikTok Optimization
1. Batch requests when possible
2. Cache frequently accessed data
3. Use appropriate video formats
4. Monitor rate limit headers
5. Implement exponential backoff

### YouTube Optimization
1. Use partial responses (fields parameter)
2. Batch API requests
3. Cache responses appropriately
4. Monitor quota usage
5. Use reports API for large datasets

### General Guidelines
1. Implement request batching
2. Use appropriate caching strategies
3. Monitor API performance
4. Optimize data transfer
5. Use compression when available

## Sample Responses and Metric Mappings

### TikTok Response
```json
{
    "video_id": "1234567890",
    "metrics": {
        "view_count": 10000,
        "share_count": 500,
        "like_count": 2000,
        "comment_count": 300,
        "engagement_rate": 0.28
    }
}
```

### YouTube Response
```json
{
    "video_id": "abcdef123",
    "metrics": {
        "views": 50000,
        "watch_time_minutes": 150000,
        "average_view_duration": 180,
        "engagement_rate": 0.15
    }
}
```

### Metric Mappings
```python
METRIC_MAPPINGS = {
    'tiktok': {
        'views': 'view_count',
        'shares': 'share_count',
        'likes': 'like_count',
        'comments': 'comment_count'
    },
    'youtube': {
        'views': 'viewCount',
        'watch_time': 'estimatedMinutesWatched',
        'average_duration': 'averageViewDuration',
        'engagement': 'engagementRate'
    }
}
```

# Platform Integration Guide

## Overview
This guide details how to integrate with supported social media platforms (Instagram, TikTok, YouTube) and implement new platform integrations.

## Supported Platforms

### Instagram Integration

#### Authentication
```python
from core.platforms import InstagramClient

client = InstagramClient(
    client_id="your_client_id",
    client_secret="your_client_secret",
    redirect_uri="your_redirect_uri"
)

# Initialize OAuth2 flow
auth_url = client.get_authorization_url()
# User visits auth_url and grants access
token = client.get_access_token(code="authorization_code")
```

#### Rate Limits
```yaml
instagram_limits:
  user_profile: 200/hour
  media_metrics: 500/hour
  story_metrics: 200/hour
  business_discovery: 30/hour
```

#### Required Permissions
- `instagram_basic`
- `instagram_manage_insights`
- `pages_read_engagement`
- `pages_show_list`

#### Metric Collection
```python
# Collect engagement metrics
engagement = client.get_media_insights(media_id)

# Collect story metrics
story_metrics = client.get_story_insights(story_id)

# Collect profile metrics
profile_metrics = client.get_account_insights()
```

### TikTok Integration

#### Authentication
```python
from core.platforms import TikTokClient

client = TikTokClient(
    app_id="your_app_id",
    app_secret="your_app_secret",
    redirect_uri="your_redirect_uri"
)

# Initialize OAuth2 flow
auth_url = client.get_authorization_url()
# User visits auth_url and grants access
token = client.get_access_token(code="authorization_code")
```

#### Rate Limits
```yaml
tiktok_limits:
  video_metrics: 1000/day
  user_metrics: 500/day
  analytics: 100/hour
```

#### Required Permissions
- `user.info.basic`
- `video.list`
- `video.info`
- `data.analytics`

#### Metric Collection
```python
# Collect video metrics
video_metrics = client.get_video_metrics(video_id)

# Collect user metrics

