    async def shutdown(self):
        """Shutdown the viral orchestration engine."""
        self.logger.info("Shutting down ViralOrchestratorEngine...")
        
        try:
            # Stop all active campaigns
            for campaign_id in list(self._active_campaigns.keys()):
                self.logger.info(f"Stopping campaign {campaign_id}")
                await self.stop_campaign(campaign_id)
            
            # Clean up AI models and resources
            self._content_optimizer = None
            self._trend_analyzer = None
            self._viral_predictor = None
            
            # Close platform connections
            self.logger.info("Closing platform connections...")
            for platform in ["instagram", "tiktok", "twitter", "youtube", "facebook", "reddit", "linkedin"]:
                await self.platform_manager.shutdown_platform(platform)
            
            # Shutdown orchestrator
            await self.orchestrator.shutdown()
            
            # Final cleanup
            self._active_campaigns.clear()
            self._viral_metrics.clear()
            self._platform_performance.clear()
            self._content_effectiveness.clear()
            
            # Update final metrics
            self._active_campaigns_gauge.set(0)
            
            self.logger.info("ViralOrchestratorEngine shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self._platform_errors.inc()
            raise
