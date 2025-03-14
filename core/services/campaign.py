import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union, Callable
from uuid import UUID

from core.automation.optimizer import AutomationOptimizer
from core.integration.coordinator import SystemCoordinator
from core.metrics.aggregator import MetricsAggregator
from core.ml.predictor import MLPredictor
from core.schemas.campaign import (
    Campaign,
    CampaignCreate,
    CampaignOptimization,
    CampaignPerformance,
    CampaignStatus,
    CampaignUpdate,
    ContentCreate,
    ContentOptimization,
    ContentUpdate,
    PlatformConfig,
    ViralMetrics,
)


class CampaignService:
    """Service for managing viral marketing campaigns"""

    def __init__(
        self,
        coordinator: SystemCoordinator,
        optimizer: AutomationOptimizer,
        predictor: MLPredictor,
        metrics: MetricsAggregator,
    ):
        self.coordinator = coordinator
        self.optimizer = optimizer
        self.predictor = predictor
        self.metrics = metrics

    async def create_campaign(self, campaign: CampaignCreate) -> Campaign:
        """Create a new viral marketing campaign with optimized initial settings"""
        # Predict viral potential and optimize initial settings
        viral_prediction = await self.predictor.predict_viral_potential(
            campaign.content_strategy,
            campaign.platforms,
            campaign.viral_goals
        )

        # Optimize campaign parameters based on prediction
        optimized_params = await self.optimizer.optimize_campaign_parameters(
            campaign,
            viral_prediction
        )

        # Initialize campaign with optimized settings
        campaign_data = {
            **campaign.dict(),
            "status": CampaignStatus.DRAFT,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "current_metrics": await self._initialize_metrics(),
            "optimization_history": [],
            "platform_performance": {},
            "viral_growth_curve": [],
            "active_experiments": [],
            "next_optimization": datetime.utcnow(),
            **optimized_params
        }

        # Create campaign through system coordinator
        return await self.coordinator.create_campaign(campaign_data)

    async def _initialize_metrics(self) -> ViralMetrics:
        """Initialize viral metrics tracking"""
        return ViralMetrics(
            viral_coefficient=0.0,
            growth_rate=0.0,
            share_rate=0.0,
            network_reach=0,
            viral_cycle_time=0.0,
            engagement_depth=0.0,
            viral_triggers=[],
            conversion_rate=0.0
        )

    async def get_campaign(self, campaign_id: UUID) -> Optional[Campaign]:
        """Get campaign details with current performance metrics"""
        campaign = await self.coordinator.get_campaign(campaign_id)
        if campaign:
            # Update metrics in real-time
            campaign.current_metrics = await self.metrics.get_current_metrics(campaign_id)
            campaign.viral_growth_curve = await self.metrics.get_growth_curve(campaign_id)
            campaign.platform_performance = await self.metrics.get_platform_metrics(campaign_id)
        return campaign

    async def update_campaign(self, campaign_id: UUID, update: CampaignUpdate) -> Campaign:
        """Update campaign with optimized changes"""
        current = await self.get_campaign(campaign_id)
        if not current:
            raise ValueError("Campaign not found")

        # Predict impact of changes
        impact_prediction = await self.predictor.predict_update_impact(
            current,
            update
        )

        # Optimize updates based on prediction
        optimized_updates = await self.optimizer.optimize_campaign_updates(
            current,
            update,
            impact_prediction
        )

        # Apply optimized updates
        campaign_data = {
            **current.dict(),
            **optimized_updates,
            "updated_at": datetime.utcnow(),
        }

        return await self.coordinator.update_campaign(campaign_id, campaign_data)

    async def archive_campaign(self, campaign_id: UUID) -> None:
        """Archive campaign and store final analytics"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        # Generate final performance report
        final_metrics = await self.metrics.get_final_metrics(campaign_id)
        
        # Store historical data
        await self.coordinator.archive_campaign(
            campaign_id,
            final_metrics
        )

        # Update status to archived
        await self.coordinator.update_campaign_status(
            campaign_id,
            CampaignStatus.ARCHIVED
        )

    async def list_campaigns(
        self,
        offset: int = 0,
        limit: int = 10,
        status: Optional[CampaignStatus] = None,
        platform: Optional[str] = None,
    ) -> List[Campaign]:
        """List campaigns with filters and real-time metrics"""
        campaigns = await self.coordinator.list_campaigns(offset, limit, status, platform)
        
        # Enrich with current metrics
        for campaign in campaigns:
            campaign.current_metrics = await self.metrics.get_current_metrics(campaign.id)
            campaign.viral_growth_curve = await self.metrics.get_growth_curve(campaign.id)

        return campaigns

    async def start_campaign_optimization(
        self,
        campaign_id: UUID,
    ) -> CampaignOptimization:
        """Start campaign optimization process"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        # Create optimization session
        optimization = await self.optimizer.create_optimization_session(
            campaign,
            campaign.optimization_types
        )

        # Update campaign status
        await self.coordinator.update_campaign_status(
            campaign_id,
            CampaignStatus.OPTIMIZING
        )

        return optimization

    async def run_optimization(
        self,
        campaign_id: UUID,
        optimization_id: UUID
    ) -> None:
        """Run campaign optimization in background"""
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                raise ValueError("Campaign not found")

            # Get current performance metrics
            current_metrics = await self.metrics.get_current_metrics(campaign_id)
            
            # Run optimization steps
            optimization_steps = [
                self.optimizer.optimize_content_viral_potential,
                self.optimizer.optimize_platform_distribution,
                self.optimizer.optimize_timing_and_frequency,
                self.optimizer.optimize_viral_triggers,
                self.optimizer.optimize_audience_targeting
            ]

            # Execute optimization pipeline
            for step in optimization_steps:
                step_result = await step(campaign, current_metrics)
                await self._apply_optimization_results(campaign_id, step_result)
                
                # Update metrics after each step
                current_metrics = await self.metrics.get_current_metrics(campaign_id)

            # Run final performance prediction
            performance_prediction = await self.predictor.predict_performance(
                campaign,
                current_metrics
            )

            # Update optimization status
            await self.coordinator.update_optimization_status(
                campaign_id,
                optimization_id,
                completed=True,
                results=performance_prediction
            )

            # Return to active status
            await self.coordinator.update_campaign_status(
                campaign_id,
                CampaignStatus.ACTIVE
            )

        except Exception as e:
            # Log error and update status
            await self.coordinator.log_optimization_error(campaign_id, optimization_id, str(e))
            await self.coordinator.update_campaign_status(
                campaign_id,
                CampaignStatus.ACTIVE
            )
            raise

    async def _apply_optimization_results(
        self,
        campaign_id: UUID,
        results: Dict
    ) -> None:
        """Apply optimization results to campaign"""
        await self.coordinator.update_campaign_parameters(campaign_id, results)
        await self.metrics.record_optimization_impact(campaign_id, results)

    async def start_ab_test(
        self,
        campaign_id: UUID,
        test_config: Dict
    ) -> UUID:
        """Start A/B testing for campaign optimization"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        # Create test variations
        variations = await self.optimizer.create_test_variations(
            campaign,
            test_config
        )

        # Initialize test tracking
        test_id = await self.coordinator.create_ab_test(
            campaign_id,
            variations,
            test_config
        )

        # Start performance monitoring
        await self.metrics.start_test_tracking(test_id, variations)

        return test_id

    async def optimize_content(
        self,
        campaign_id: UUID,
        content: ContentCreate
    ) -> ContentOptimization:
        """Optimize content for viral potential"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        # Analyze viral potential
        viral_potential = await self.predictor.analyze_content_potential(
            content,
            campaign.platforms
        )

        # Generate optimization suggestions
        optimization = await self.optimizer.optimize_content(
            content,
            viral_potential,
            campaign.viral_goals
        )

        # Apply platform-specific optimizations
        for platform in campaign.platforms:
            platform_optimization = await self.optimizer.optimize_for_platform(
                optimization.content,
                platform,
                viral_potential
            )
            optimization.platform_variants[platform] = platform_optimization

        return optimization

    async def schedule_content(
        self,
        campaign_id: UUID,
        content_id: UUID,
        platforms: List[str]
    ) -> Dict[str, datetime]:
        """Schedule optimized content distribution"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        # Get platform-specific metrics
        platform_metrics = await self.metrics.get_platform_metrics(campaign_id)

        # Calculate optimal timing
        schedule = await self.optimizer.calculate_optimal_timing(
            campaign,
            content_id,
            platforms,
            platform_metrics
        )

        # Create distribution tasks
        for platform, timing in schedule.items():
            await self.coordinator.schedule_content_distribution(
                campaign_id,
                content_id,
                platform,
                timing
            )

        return schedule

    async def sync_platforms(
        self,
        campaign_id: UUID,
        platforms: List[PlatformConfig]
    ) -> Dict[str, bool]:
        """Synchronize campaign across platforms"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        sync_status = {}
        for platform in platforms:
            try:
                # Sync platform content and settings
                await self.coordinator.sync_platform(
                    campaign_id,
                    platform
                )
                
                # Verify sync status
                status = await self.coordinator.verify_platform_sync(
                    campaign_id,
                    platform.platform_id
                )
                
                sync_status[platform.platform_id] = status
            except Exception as e:
                sync_status[platform.platform_id] = False
                await self.coordinator.log_sync_error(campaign_id, platform.platform_id, str(e))

        return sync_status

    async def get_performance_analytics(
        self,
        campaign_id: UUID
    ) -> CampaignPerformance:
        """Get comprehensive performance analytics"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        # Gather all metrics
        current_metrics = await self.metrics.get_current_metrics(campaign_id)
        growth_curve = await self.metrics.get_growth_curve(campaign_id)
        platform_metrics = await self.metrics.get_platform_metrics(campaign_id)
        roi_metrics = await self.metrics.calculate_roi(campaign_id)

        # Generate predictions
        predictions = await self.predictor.predict_future_performance(
            campaign,
            current_metrics,
            growth_curve
        )

        # Calculate trend analysis
        trends = await self.metrics.analyze_trends(
            campaign_id,
            growth_curve,
            platform_metrics
        )

        return CampaignPerformance(
            current_metrics=current_metrics,
            growth_curve=growth_curve,
            platform_metrics=platform_metrics,
            roi_metrics=roi_metrics,
            predictions=predictions,
            trends=trends
        )

    async def monitor_performance(
        self,
        campaign_id: UUID,
        callback: Optional[callable] = None
    ) -> None:
        """Start real-time performance monitoring"""
        async def _monitor():
            while True:
                try:
                    # Get current metrics
                    metrics = await self.metrics.get_current_metrics(campaign_id)
                    
                    # Check for triggers
                    triggers = await self.optimizer.check_optimization_triggers(
                        campaign_id,
                        metrics
                    )

                    if triggers:
                        # Auto-optimize if triggers detected
                        optimization = await self.start_campaign_optimization(campaign_id)
                        await self.run_optimization(campaign_id, optimization.id)

                    if callback:
                        await callback(metrics)

                except Exception as e:
                    await self.coordinator.log_monitoring_error(campaign_id, str(e))

                await asyncio.sleep(60)  # Monitor interval

        # Start monitoring task
        monitoring_task = asyncio.create_task(_monitor())
        await self.coordinator.register_monitoring_task(campaign_id, monitoring_task)
