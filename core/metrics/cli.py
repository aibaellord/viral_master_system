import click
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional
import logging

from .config import MetricsConfig
from .factory import MetricsServiceFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_metrics(metrics):
    """Format metrics for CLI output."""
    return {
        "timestamp": metrics.timestamp.isoformat(),
        "total_reach": metrics.total_reach,
        "total_impressions": metrics.total_impressions,
        "total_shares": metrics.total_shares,
        "average_engagement_rate": f"{metrics.average_engagement_rate:.2f}%",
        "cross_platform_viral_coefficient": f"{metrics.cross_platform_viral_coefficient:.2f}",
        "platform_breakdown": {
            name: {
                "engagement_rate": f"{m.engagement_rate:.2f}%",
                "viral_coefficient": f"{m.viral_coefficient:.2f}",
                "reach": m.reach,
                "impressions": m.impressions,
                "shares": m.shares
            }
            for name, m in metrics.platform_breakdown.items()
        }
    }

@click.group()
def cli():
    """Metrics Aggregator CLI"""
    pass

@cli.command()
@click.option('--json-output', is_flag=True, help="Output in JSON format")
async def collect(json_output):
    """Collect current metrics from all platforms."""
    try:
        factory = MetricsServiceFactory()
        aggregator = factory.create_metrics_aggregator()
        
        metrics = await aggregator.get_aggregated_metrics()
        formatted_metrics = format_metrics(metrics)
        
        if json_output:
            click.echo(json.dumps(formatted_metrics, indent=2))
        else:
            click.echo("\nMetrics Summary:")
            click.echo(f"Timestamp: {formatted_metrics['timestamp']}")
            click.echo(f"Total Reach: {formatted_metrics['total_reach']:,}")
            click.echo(f"Total Impressions: {formatted_metrics['total_impressions']:,}")
            click.echo(f"Average Engagement Rate: {formatted_metrics['average_engagement_rate']}")
            click.echo(f"Cross-Platform Viral Coefficient: {formatted_metrics['cross_platform_viral_coefficient']}")
            
            click.echo("\nPlatform Breakdown:")
            for platform, stats in formatted_metrics['platform_breakdown'].items():
                click.echo(f"\n{platform.upper()}:")
                for metric, value in stats.items():
                    click.echo(f"  {metric}: {value}")
    
    except Exception as e:
        logger.error(f"Error collecting metrics: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--days', default=7, help="Number of days to analyze")
@click.option('--json-output', is_flag=True, help="Output in JSON format")
async def trends(days, json_output):
    """Analyze metric trends over time."""
    try:
        factory = MetricsServiceFactory()
        aggregator = factory.create_metrics_aggregator()
        
        # Get historical metrics
        metrics_history = []  # You would implement this to fetch from your storage
        
        trends = aggregator.perform_trend_analysis(metrics_history)
        
        if json_output:
            click.echo(json.dumps(trends, indent=2))
        else:
            click.echo("\nTrend Analysis:")
            for metric, growth in trends.items():
                click.echo(f"{metric}: {growth:+.2f}%")
    
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--interval', default=900, help="Collection interval in seconds")
async def monitor(interval):
    """Start continuous monitoring of metrics."""
    try:
        factory = MetricsServiceFactory()
        aggregator = factory.create_metrics_aggregator()
        
        click.echo(f"Starting metrics monitor (interval: {interval}s)")
        click.echo("Press Ctrl+C to stop")
        
        while True:
            try:
                metrics

