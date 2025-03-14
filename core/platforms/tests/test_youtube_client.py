import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from ..youtube_client import YouTubeClient, YouTubeMetrics, RateLimitExceeded, AuthenticationError, ApiError

@pytest.fixture
def youtube_client():
    client = YouTubeClient(
        client_id="test_client_id",
        client_secret="test_client_secret",
        token_file="test_token.json",
        credentials_file="test_credentials.json"
    )
    return client

@pytest.fixture
def mock_youtube_service():
    with patch('googleapiclient.discovery.build') as mock_build:
        yield mock_build

@pytest.mark.asyncio
async def test_authentication_success(youtube_client, mock_youtube_service):
    """Test successful authentication flow."""
    with patch.object(youtube_client, '_load_credentials') as mock_load:
        mock_credentials = Mock()
        mock_credentials.valid = True
        mock_load.return_value = mock_credentials
        
        await youtube_client.authenticate()
        
        assert youtube_client.credentials == mock_credentials
        assert youtube_client.youtube is not None
        assert youtube_client.youtube_analytics is not None

@pytest.mark.asyncio
async def test_authentication_refresh(youtube_client, mock_youtube_service):
    """Test credentials refresh when expired."""
    with patch.object(youtube_client, '_load_credentials') as mock_load:
        mock_credentials = Mock()
        mock_credentials.valid = False
        mock_credentials.expired = True
        mock_credentials.refresh_token = True
        mock_load.return_value = mock_credentials
        
        await youtube_client.authenticate()
        
        mock_credentials.refresh.assert_called_once()

@pytest.mark.asyncio
async def test_authentication_failure(youtube_client):
    """Test authentication failure handling."""
    with patch.object(youtube_client, '_load_credentials', side_effect=Exception("Auth failed")):
        with pytest.raises(AuthenticationError):
            await youtube_client.authenticate()

@pytest.mark.asyncio
async def test_collect_metrics_success(youtube_client):
    """Test successful metrics collection."""
    # Mock API responses
    mock_video_response = {
        'items': [{
            'statistics': {
                'viewCount': '1000',
                'likeCount': '100',
                'commentCount': '50'
            }
        }]
    }
    
    mock_analytics_data = {
        'estimatedMinutesWatched': 5000.0,
        'averageViewDuration': 120.0,
        'audienceRetentionRate': 75.0,
        'shares': 30,
        'subscribersGained': 20,
        'subscribersLost': 5,
        'monetizedPlaybacks': 800,
        'estimatedRevenue': 50.0
    }
    
    with patch.object(youtube_client, '_make_request') as mock_request:
        mock_request.side_effect = [
            mock_video_response,
            {'rows': [[1] + [val for val in mock_analytics_data.values()]]},
            {'rows': [[0, 0.8], [1, 0.7]]}  # Retention data
        ]
        
        metrics = await youtube_client.collect_metrics('test_video_id')
        
        assert isinstance(metrics, YouTubeMetrics)
        assert metrics.views == 1000
        assert metrics.likes == 100
        assert metrics.comments == 50
        assert metrics.watch_time_minutes == 5000.0
        assert metrics.engagement_rate == 15.0  # (100 + 50) / 1000 * 100

@pytest.mark.asyncio
async def test_quota_limit_exceeded(youtube_client):
    """Test quota limit handling."""
    youtube_client._quota_used = youtube_client.MAX_DAILY_QUOTA
    
    with pytest.raises(RateLimitExceeded):
        await youtube_client.collect_metrics('test_video_id')

@pytest.mark.asyncio
async def test_quota_reset(youtube_client):
    """Test quota reset functionality."""
    youtube_client._quota_used = youtube_client.MAX_DAILY_QUOTA
    youtube_client._quota_reset_time = datetime.now() - timedelta(seconds=1)
    
    # Mock API responses
    with patch.object(youtube_client, '_make_request') as mock_request:
        mock_request.return_value = {
            'items': [{
                'statistics': {
                    'viewCount': '1000',
                    'likeCount': '100',
                    'commentCount': '50'
                }
            }]
        }
        
        # Should not raise RateLimitExceeded
        await youtube_client.collect_metrics('test_video_id')
        assert youtube_client._quota_used < youtube_client.MAX_DAILY_QUOTA

@pytest.mark.asyncio
async def test_api_error_handling(youtube_client):
    """Test API error handling."""
    with patch.object(youtube_client, '_make_request', side_effect=ApiError("API Error")):
        with pytest.raises(ApiError):
            await youtube_client.collect_metrics('test_video_id')

@pytest.mark.asyncio
async def test_retention_data_calculation(youtube_client):
    """Test audience retention calculation."""
    mock_retention_response = {
        'rows': [
            [0.0, 1.0],
            [0.25, 0.9],
            [0.5, 0.8],
            [0.75, 0.7],
            [1.0, 0.6]
        ]
    }
    
    with patch.object(youtube_client, '_make_request') as mock_request:
        mock_request.return_value = mock_retention_response
        
        retention_data = await youtube_client._get_audience_retention('test_video_id')
        assert retention_data['average

