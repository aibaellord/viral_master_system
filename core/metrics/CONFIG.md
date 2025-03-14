# Metrics System Configuration Guide

## Overview

The metrics system uses a flexible configuration system that supports both YAML configuration files and environment variables. This document describes all available configuration options and how to use them.

## Configuration Sources

Configurations can be specified in three ways, in order of precedence:

1. Environment variables (highest precedence)
2. YAML configuration file
3. Default values (lowest precedence)

## Configuration Sections

### Database Configuration

Controls the connection to the metrics database.

```yaml
database:
  host: localhost          # Database server hostname
  port: 5432              # Database server port
  name: metrics_db        # Database name
  pool_size: 10          # Connection pool size
  max_overflow: 20       # Maximum number of connections
  timeout: 30            # Connection timeout in seconds
```

Environment variables:
- METRICS_DB_USER
- METRICS_DB_PASSWORD
- METRICS_DB_HOST
- METRICS_DB_PORT
- METRICS_DB_NAME

### Cache Configuration

Controls the caching behavior for metric data.

```yaml
cache:
  strategy: redis         # Options: memory, redis, none
  ttl: 300               # Cache TTL in seconds
  max_size: 1000         # Maximum cache entries
  redis_url: redis://localhost:6379/0
```

Environment variables:
- REDIS_URL

### Platform Configuration

Platform-specific API settings for each supported social media platform.

```yaml
platforms:
  instagram:
    rate_limit: 100      # API rate limit per minute
    batch_size: 50       # Batch size for API requests
    timeout: 30          # API timeout in seconds
    retry_attempts: 3    # Number of retry attempts
    retry_delay: 5       # Delay between retries in seconds
```

Environment variables (per platform):
- {PLATFORM}_API_KEY
- {PLATFORM}_API_SECRET
- {PLATFORM}_RATE_LIMIT
- {PLATFORM}_BATCH_SIZE

### Alert Thresholds

Configures thresholds for various metrics that trigger alerts.

```yaml
alert_thresholds:
  viral_coefficient: 1.5  # Viral coefficient threshold
  engagement_rate: 0.05   # Minimum engagement rate
  growth_rate: 0.1        # Minimum growth rate
  sentiment_score: 0.7    # Minimum sentiment score
  alert_interval: 300     # Alert check interval in seconds
```

Environment variables:
- VIRAL_COEFFICIENT_THRESHOLD
- ENGAGEMENT_RATE_THRESHOLD
- GROWTH_RATE_THRESHOLD
- SENTIMENT_SCORE_THRESHOLD

### Retention Policy

Controls data retention and backup settings.

```yaml
retention:
  raw_data_days: 30           # Days to keep raw data
  aggregated_data_days: 365   # Days to keep aggregated data
  snapshot_interval: 300      # Snapshot interval in seconds
  compression_enabled: true   # Enable data compression
  backup_enabled: true        # Enable automated backups
  backup_interval: 86400      # Backup interval in seconds
```

### Performance Settings

Controls system performance and resource utilization.

```yaml
performance:
  worker_threads: 4          # Number of worker threads
  batch_size: 1000          # Processing batch size
  queue_size: 5000          # Maximum queue size
  processing_timeout: 60     # Processing timeout in seconds
  metrics_buffer_size: 10000 # Metrics buffer size
```

Environment variables:
- WORKER_THREADS
- BATCH_SIZE
- QUEUE_SIZE

### Export Configuration

Controls data export settings.

```yaml
export:
  formats:                   # Supported export formats
    - json
    - csv
  compression: true         # Enable export compression
  max_rows: 1000000        # Maximum rows per export
  chunk_size: 10000        # Export chunk size
  include_metadata: true    # Include metadata in exports
```

## Usage Example

1. Copy `config.example.yaml` to `config.yaml`
2. Copy `config.env.example` to `.env`
3. Modify the configurations as needed
4. Load the configuration in your code:

```python
from core.metrics.config import MetricsConfig

# Load config from file
config = MetricsConfig.load_config("config.yaml")

# Access configuration values
db_host = config.database.host
cache_ttl = config.cache.ttl
```

## Best Practices

1. Never commit API keys or secrets to version control
2. Use environment variables for sensitive information
3. Use YAML configuration for development and testing
4. Regularly review and update alert thresholds
5. Monitor performance settings in production

## Troubleshooting

Common issues and solutions:

1. Database connection issues:
   - Verify database credentials
   - Check database host and port
   - Ensure database is running

2. Cache issues:
   - Verify Redis URL if using Redis
   - Check cache TTL settings
   - Monitor cache size

3. Platform API issues:
   - Verify API keys and secrets
   - Check rate limits
   - Monitor API timeout settings

