import pytest
import asyncio
from datetime import datetime, timedelta

from core.automation.viral_engine import ViralLoopEngine
from core.automation.optimizer import AutomationOptimizer
from core.ml.predictor import MLPredictor
from core.services.campaign import CampaignService
from core.integration.coordinator import SystemCoordinator
from core.platforms.instagram_client import InstagramClient
from core.platforms.tiktok_client import TikTokClient
from core.platforms.youtube_client import YouTubeClient
from core.metrics.aggregator import MetricsAggregator
from core.analytics.analyzer import AnalyticsEngine

@pytest.fixture
async def system_components():
    """Initialize all system components for testing."""
    viral_engine = ViralLoopEngine()
    optimizer = AutomationOptimizer()
    ml_predictor = MLPredictor()
    metrics = MetricsAggregator()
    analytics = AnalyticsEngine()
    
    # Initialize platform clients with test credentials
    instagram = InstagramClient(test_mode=True)
    tiktok = TikTokClient(test_mode=True)
    youtube = YouTubeClient(test_mode=True)
    
    campaign_service = CampaignService(
        viral_engine=viral_engine,
        optimizer=optimizer,
        predictor=ml_predictor,
        metrics=metrics,
        platforms=[instagram, tiktok, youtube]
    )
    
    coordinator = SystemCoordinator(
        viral_engine=viral_engine,
        optimizer=optimizer,
        predictor=ml_predictor,
        campaign_service=campaign_service,
        metrics=metrics,
        analytics=analytics
    )
    
    await coordinator.initialize()
    
    yield {
        'coordinator': coordinator,
        'viral_engine': viral_engine,
        'optimizer': optimizer,
        'predictor': ml_predictor,
        'campaign_service': campaign_service,
        'metrics': metrics,
        'analytics': analytics,
        'platforms': {
            'instagram': instagram,
            'tiktok': tiktok,
            'youtube': youtube
        }
    }
    
    await coordinator.shutdown()

@pytest.mark.asyncio
async def test_end_to_end_campaign_flow(system_components):
    """Test complete campaign lifecycle with optimization."""
    coordinator = system_components['coordinator']
    campaign_service = system_components['campaign_service']
    
    # Create campaign
    campaign = await campaign_service.create_campaign({
        'name': 'Test Viral Campaign',
        'objective': 'VIRAL_GROWTH',
        'budget': 1000,
        'platforms': ['instagram', 'tiktok', 'youtube'],
        'optimization_settings': {
            'viral_coefficient_target': 2.0,
            'auto_optimize': True
        }
    })
    
    # Add test content
    content_id = await campaign_service.add_content(campaign.id, {
        'type': 'VIDEO',
        'title': 'Viral Test Content',
        'description': 'Test content for viral optimization',
        'platforms': {
            'instagram': {'format': 'REEL'},
            'tiktok': {'format': 'TIKTOK_VIDEO'},
            'youtube': {'format': 'SHORTS'}
        }
    })
    
    # Wait for initial optimization
    await asyncio.sleep(2)
    
    # Verify optimization occurred
    optimization_status = await coordinator.get_optimization_status(campaign.id)
    assert optimization_status.is_optimized
    assert optimization_status.viral_coefficient > 1.0
    
    # Simulate performance data
    await campaign_service.simulate_metrics(campaign.id, {
        'views': 10000,
        'shares': 2000,
        'engagement_rate': 0.15,
        'viral_coefficient': 1.8
    })
    
    # Verify auto-optimization response
    await asyncio.sleep(2)
    updated_status = await coordinator.get_optimization_status(campaign.id)
    assert updated_status.optimization_count > optimization_status.optimization_count
    
    # Test cross-platform synchronization
    sync_status = await coordinator.get_platform_sync_status(campaign.id)
    assert all(platform.is_synced for platform in sync_status.values())

@pytest.mark.asyncio
async def test_viral_loop_optimization(system_components):
    """Test viral loop detection and optimization."""
    viral_engine = system_components['viral_engine']
    optimizer = system_components['optimizer']
    
    # Create test viral loop configuration
    loop_config = {
        'trigger_type': 'SHARE',
        'audience_segment': 'EARLY_ADOPTERS',
        'viral_mechanics': ['SOCIAL_PROOF', 'FOMO', 'RECIPROCITY'],
        'platforms': ['instagram', 'tiktok']
    }
    
    # Initialize viral loop
    loop_id = await viral_engine.create_viral_loop(loop_config)
    
    # Simulate viral activities
    for _ in range(5):
        await viral_engine.simulate_viral_action(loop_id, {
            'action': 'SHARE',
            'platform': 'instagram',
            'reach': 1000,
            'conversions': 200
        })
    
    # Wait for optimization
    await asyncio.sleep(2)
    
    # Verify optimization occurred
    loop_metrics = await viral_engine.get_loop_metrics(loop_id)
    assert loop_metrics.viral_coefficient > 1.5
    assert loop_metrics.optimization_count > 0
    
    # Verify optimizer recommendations
    recommendations = await optimizer.get_loop_recommendations(loop_id)
    assert len(recommendations) > 0
    assert all(r.confidence_score > 0.8 for r in recommendations)

@pytest.mark.asyncio
async def test_ai_decision_making(system_components):
    """Test AI-driven decision making and adaptations."""
    predictor = system_components['predictor']
    optimizer = system_components['optimizer']
    
    # Initialize test scenario
    scenario = {
        'content_type': 'VIDEO',
        'platform': 'tiktok',
        'audience': 'GEN_Z',
        'historical_performance': {
            'viral_coefficient_mean': 1.5,
            'engagement_rate_mean': 0.12,
            'share_rate_mean': 0.08
        }
    }
    
    # Get AI predictions
    predictions = await predictor.predict_performance(scenario)
    assert predictions.confidence_score > 0.85
    
    # Get optimization recommendations
    recommendations = await optimizer.get_ai_recommendations(scenario)
    assert len(recommendations) > 0
    
    # Verify adaptation to new data
    new_data = {
        'viral_coefficient': 2.0,
        'engagement_rate': 0.18,
        'share_rate': 0.11
    }
    await predictor.update_model(scenario, new_data)
    
    # Verify model adaptation
    updated_predictions = await predictor.predict_performance(scenario)
    assert updated_predictions.confidence_score > predictions.confidence_score

@pytest.mark.asyncio
async def test_system_resilience(system_components):
    """Test system's ability to handle failures and recover."""
    coordinator = system_components['coordinator']
    campaign_service = system_components['campaign_service']
    
    # Create test campaign
    campaign = await campaign_service.create_campaign({
        'name': 'Resilience Test Campaign',
        'platforms': ['instagram', 'tiktok', 'youtube']
    })
    
    # Simulate platform failure
    await system_components['platforms']['instagram'].simulate_failure()
    
    # Verify system continues with other platforms
    status = await coordinator.get_system_status()
    assert status.is_operational
    assert not status.platform_status['instagram'].is_operational
    assert status.platform_status['tiktok'].is_operational
    
    # Verify automatic retry and recovery
    await asyncio.sleep(5)
    recovered_status = await coordinator.get_system_status()
    assert recovered_status.platform_status['instagram'].is_operational
    
    # Test data consistency
    metrics_before = await campaign_service.get_campaign_metrics(campaign.id)
    
    # Simulate coordinator failure and recovery
    await coordinator.simulate_failure()
    await asyncio.sleep(2)
    await coordinator.recover()
    
    # Verify data consistency
    metrics_after = await campaign_service.get_campaign_metrics(campaign.id)
    assert metrics_before == metrics_after

@pytest.mark.benchmark
async def test_system_performance(system_components):
    """Benchmark system performance under load."""
    campaign_service = system_components['campaign_service']
    
    # Measure campaign creation performance
    async def create_campaigns(count):
        tasks = []
        for i in range(count):
            tasks.append(campaign_service.create_campaign({
                'name': f'Performance Test Campaign {i}',
                'platforms': ['instagram', 'tiktok', 'youtube']
            }))
        return await asyncio.gather(*tasks)
    
    # Benchmark campaign creation
    start_time = datetime.now()
    campaigns = await create_campaigns(100)
    duration = datetime.now() - start_time
    
    assert len(campaigns) == 100
    assert duration < timedelta(seconds=5)
    
    # Test optimization performance
    start_time = datetime.now()
    await asyncio.gather(*[
        campaign_service.optimize_campaign(c.id)
        for c in campaigns[:10]  # Test with first 10 campaigns
    ])
    optimization_duration = datetime.now() - start_time
    
    assert optimization_duration < timedelta(seconds=10)

