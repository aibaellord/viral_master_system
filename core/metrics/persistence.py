from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import json

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

from .types import MetricType, PlatformType, TimeSeriesMetric
from .config import MetricsConfig

Base = declarative_base()

class MetricRecord(Base):
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String, nullable=False)
    metric_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float)
    metadata = Column(JSONB)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_metrics_platform_type_ts', 'platform', 'metric_type', 'timestamp'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )

class TimeSeriesRecord(Base):
    __tablename__ = 'time_series'
    
    id = Column(Integer, primary_key=True)
    metric_id = Column(Integer, ForeignKey('metrics.id'))
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    
    metric = relationship("MetricRecord")
    
    __table_args__ = (
        Index('idx_timeseries_metric_ts', 'metric_id', 'timestamp'),
    )

class MetricAggregation(Base):
    __tablename__ = 'metric_aggregations'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    aggregation_type = Column(String, nullable=False)  # e.g., 'avg', 'sum', 'min', 'max'
    value = Column(Float, nullable=False)
    metadata = Column(JSONB)
    
    __table_args__ = (
        Index('idx_aggregation_type_period', 'metric_type', 'period_start', 'period_end'),
    )

class MetricVersion(Base):
    __tablename__ = 'metric_versions'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    schema = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_metric_version', 'metric_type', 'version', unique=True),
    )

class MetricsPersistence:
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.engine = create_engine(config.database.url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
    def store_metric(self, platform: PlatformType, metric_type: MetricType, 
                     value: float, timestamp: datetime, metadata: Optional[Dict] = None) -> int:
        """Store a single metric value with optional metadata."""
        try:
            with self.Session() as session:
                metric = MetricRecord(
                    platform=platform.value,
                    metric_type=metric_type.value,
                    timestamp=timestamp,
                    value=value,
                    metadata=metadata
                )
                session.add(metric)
                session.commit()
                return metric.id
        except SQLAlchemyError as e:
            raise Exception(f"Failed to store metric: {str(e)}")

    def store_time_series(self, metric_id: int, 
                         data_points: List[TimeSeriesMetric]) -> None:
        """Store multiple time series data points for a metric."""
        try:
            with self.Session() as session:
                series_records = [
                    TimeSeriesRecord(
                        metric_id=metric_id,
                        timestamp=point.timestamp,
                        value=point.value
                    )
                    for point in data_points
                ]
                session.bulk_save_objects(series_records)
                session.commit()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to store time series: {str(e)}")

    def aggregate_metrics(self, metric_type: MetricType, 
                         start_time: datetime, end_time: datetime,
                         aggregation_type: str) -> float:
        """Calculate metric aggregations for a given time period."""
        try:
            with self.Session() as session:
                query = session.query(func.avg(MetricRecord.value).label('avg_value')).\
                    filter(
                        MetricRecord.metric_type == metric_type.value,
                        MetricRecord.timestamp.between(start_time, end_time)
                    )
                
                if aggregation_type == 'sum':
                    query = query.with_entities(func.sum(MetricRecord.value).label('sum_value'))
                elif aggregation_type == 'min':
                    query = query.with_entities(func.min(MetricRecord.value).label('min_value'))
                elif aggregation_type == 'max':
                    query = query.with_entities(func.max(MetricRecord.value).label('max_value'))
                
                result = query.scalar()
                return result or 0.0
        except SQLAlchemyError as e:
            raise Exception(f"Failed to aggregate metrics: {str(e)}")

    def query_metrics(self, platform: Optional[PlatformType] = None,
                     metric_type: Optional[MetricType] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Query metrics with optional filtering criteria."""
        try:
            with self.Session() as session:
                query = session.query(MetricRecord)
                
                if platform:
                    query = query.filter(MetricRecord.platform == platform.value)
                if metric_type:
                    query = query.filter(MetricRecord.metric_type == metric_type.value)
                if start_time:
                    query = query.filter(MetricRecord.timestamp >= start_time)
                if end_time:
                    query = query.filter(MetricRecord.timestamp <= end_time)
                
                query = query.order_by(MetricRecord.timestamp.desc()).limit(limit)
                
                return [
                    {
                        'id': record.id,
                        'platform': record.platform,
                        'metric_type': record.metric_type,
                        'timestamp': record.timestamp,
                        'value': record.value,
                        'metadata': record.metadata
                    }
                    for record in query.all()
                ]
        except SQLAlchemyError as e:
            raise Exception(f"Failed to query metrics: {str(e)}")

    def cleanup_old_data(self, retention_days: int) -> int:
        """Remove metrics older than the specified retention period."""
        try:
            with self.Session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                deleted = session.query(MetricRecord).\
                    filter(MetricRecord.timestamp < cutoff_date).\
                    delete(synchronize_session=False)
                session.commit()
                return deleted
        except SQLAlchemyError as e:
            raise Exception(f"Failed to cleanup old data: {str(e)}")

    def backup_metrics(self, backup_path: str, start_time: Optional[datetime] = None) -> None:
        """Backup metrics data to a file."""
        try:
            with self.Session() as session:
                query = session.query(MetricRecord)
                if start_time:
                    query = query.filter(MetricRecord.timestamp >= start_time)
                
                metrics = [
                    {
                        'id': record.id,
                        'platform': record.platform,
                        'metric_type': record.metric_type,
                        'timestamp': record.timestamp.isoformat(),
                        'value': record.value,
                        'metadata': record.metadata
                    }
                    for record in query.all()
                ]
                
                with open(backup_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
        except (SQLAlchemyError, IOError) as e:
            raise Exception(f"Failed to backup metrics: {str(e)}")

    def restore_metrics(self, backup_path: str) -> int:
        """Restore metrics from a backup file."""
        try:
            with open(backup_path, 'r') as f:
                metrics = json.load(f)
            
            with self.Session() as session:
                for metric in metrics:
                    record = MetricRecord(
                        platform=metric['platform'],
                        metric_type=metric['metric_type'],
                        timestamp=datetime.fromisoformat(metric['timestamp']),
                        value=metric['value'],
                        metadata=metric['metadata']
                    )
                    session.add(record)
                session.commit()
                return len(metrics)
        except (SQLAlchemyError, IOError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to restore metrics: {str(e)}")

    def get_metric_version(self, metric_type: MetricType) -> Optional[Dict[str, Any]]:
        """Get the latest version information for a metric type."""
        try:
            with self.Session() as session:
                version = session.query(MetricVersion).\
                    filter(MetricVersion.metric_type == metric_type.value).\
                    order_by(MetricVersion.version.desc()).\
                    first()
                
                if version:
                    return {
                        'metric_type': version.metric_type,
                        'version': version.version,
                        'schema': version.schema,
                        'created_at': version.created_at
                    }
                return None
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get metric version: {str(e)}")

    def store_metric_version(self, metric_type: MetricType, 
                           schema: Dict[str, Any]) -> int:
        """Store a new version of a metric type schema."""
        try:
            with self.Session() as session:
                # Get current latest version
                current_version = session.query(func.max(MetricVersion.version)).\
                    filter(MetricVersion.metric_type == metric_type.value).\
                    scalar() or 0
                
                # Create new version
                new_version = MetricVersion(
                    metric_type=metric_type.value,
                    version=current_version + 1,
                    schema=schema
                )
                session.add(new_version)
                session.commit()
                return new_version.version
        except SQLAlchemyError as e:
            raise Exception(f"Failed to store metric version: {str(e)}")

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import csv
from io import StringIO
from functools import lru_cache
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.metrics.metrics_aggregator import MetricsSnapshot
from core.config import get_metrics_config
from core.cache import Cache

class MetricsPersistence:
    def __init__(self):
        self.db_path = Path.home() / '.viral_master_system' / 'metrics.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = Cache()
        self.batch_size = 1000
        self.init_db()

    def init_db(self):
        """Initialize the SQLite database and create necessary tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    platform TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_platform 
                ON metrics(platform)
            """)

    def store_metrics(self, metrics: MetricsSnapshot):
        """Store a metrics snapshot in the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            timestamp = datetime.now().isoformat()
            
            for platform, platform_metrics in metrics.platform_metrics.items():
                for metric_name, metric_value in platform_metrics.items():
                    conn.execute(
                        "INSERT INTO metrics (timestamp, platform, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                        (timestamp, platform, metric_name, metric_value)
                    )

    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        platform: Optional[str] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Retrieve metrics from the database within the specified time range."""
        query = """
            SELECT timestamp, platform, metric_name, metric_value
            FROM metrics
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]
        
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        
        if metric_names:
            placeholders = ','.join('?' * len(metric_names))
            query += f" AND metric_name IN ({placeholders})"
            params.extend(metric_names)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_metrics(self, platform: Optional[str] = None) -> Dict[str, float]:
        """Get the most recent metrics for a platform."""
        query = """
            SELECT m1.*
            FROM metrics m1
            INNER JOIN (
                SELECT platform, metric_name, MAX(timestamp) as max_timestamp
                FROM metrics
                WHERE platform = ? OR ? IS NULL
                GROUP BY platform, metric_name
            ) m2 
            ON m1.platform = m2.platform 
            AND m1.metric_name = m2.metric_name 
            AND m1.timestamp = m2.max_timestamp
        """
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (platform, platform))
            return {row['metric_name']: row['metric_value'] for row in cursor.fetchall()}

    def cleanup_old_metrics(self):
        """Remove metrics older than the retention period."""
        config = get_metrics_config()
        retention_days = config.get('retention_days', 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )

    def get_trend_data(
        self,
        metric_name: str,
        platform: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Union[datetime, float]]]:
        """Get trend data for a specific metric over time."""
        start_time = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT timestamp, AVG(metric_value) as value
            FROM metrics
            WHERE metric_name = ?
            AND timestamp >= ?
        """
        params = [metric_name, start_time.isoformat()]
        
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        
        query += """
            GROUP BY strftime('%Y-%m-%d %H', timestamp)
            ORDER BY timestamp ASC
        """
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def store_metrics_batch(self, metrics_list: List[MetricsSnapshot]):
        """Store multiple metrics snapshots in a single transaction for better performance."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for i in range(0, len(metrics_list), self.batch_size):
                batch = metrics_list[i:i + self.batch_size]
                values = []
                timestamp = datetime.now().isoformat()
                
                for metrics in batch:
                    for platform, platform_metrics in metrics.platform_metrics.items():
                        values.extend(
                            (timestamp, platform, name, value)
                            for name, value in platform_metrics.items()
                        )
                
                cursor.executemany(
                    "INSERT INTO metrics (timestamp, platform, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                    values
                )

    @lru_cache(maxsize=100)
    def get_performance_analytics(
        self,
        platform: str,
        metric_name: str,
        timeframe: str = '24h'
    ) -> Dict[str, float]:
        """Calculate performance analytics for a specific metric.
        
        Args:
            platform: Platform name
            metric_name: Name of the metric
            timeframe: Time window for analysis ('24h', '7d', '30d')
            
        Returns:
            Dict containing analytics (mean, median, std_dev, min, max, percentiles)
        """
        time_windows = {'24h': 24, '7d': 168, '30d': 720}
        hours = time_windows.get(timeframe, 24)
        
        data = self.get_trend_data(metric_name, platform, hours)
        if not data:
            return {}
            
        values = [d['value'] for d in data]
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'percentile_25': np.percentile(values, 25),
            'percentile_75': np.percentile(values, 75),
            'variance': np.var(values)
        }

    def aggregate_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        grouping: str = 'hourly',
        platforms: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Aggregate metrics data with various time-based groupings.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            grouping: Time grouping ('hourly', 'daily', 'weekly', 'monthly')
            platforms: Optional list of platforms to include
            
        Returns:
            DataFrame with aggregated metrics
        """
        group_formats = {
            'hourly': '%Y-%m-%d %H',
            'daily': '%Y-%m-%d',
            'weekly': '%Y-%W',
            'monthly': '%Y-%m'
        }
        
        format_str = group_formats.get(grouping, '%Y-%m-%d %H')
        
        query = """
            SELECT 
                strftime(?, timestamp) as period,
                platform,
                metric_name,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                COUNT(*) as count
            FROM metrics
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [format_str, start_time.isoformat(), end_time.isoformat()]
        
        if platforms:
            placeholders = ','.join('?' * len(platforms))
            query += f" AND platform IN ({placeholders})"
            params.extend(platforms)
        
        query += """
            GROUP BY period, platform, metric_name
            ORDER BY period, platform, metric_name
        """
        
        with sqlite3.connect(str(self.db_path)) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def compare_metrics(
        self,
        metric_name: str,
        platform: str,
        period1: Tuple[datetime, datetime],
        period2: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Compare metrics between two time periods.
        
        Args:
            metric_name: Name of the metric to compare
            platform: Platform name
            period1: (start_time, end_time) for first period
            period2: (start_time, end_time) for second period
            
        Returns:
            Dictionary containing comparison statistics
        """
        def get_period_stats(start_time, end_time):
            query = """
                SELECT 
                    AVG(metric_value) as mean,
                    MIN(metric_value) as min,
                    MAX(metric_value) as max,
                    COUNT(*) as count
                FROM metrics
                WHERE timestamp BETWEEN ? AND ?
                AND platform = ?
                AND metric_name = ?
            """
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    query,
                    (start_time.isoformat(), end_time.isoformat(), platform, metric_name)
                )
                return cursor.fetchone()

        stats1 = get_period_stats(period1[0], period1[1])
        stats2 = get_period_stats(period2[0], period2[1])
        
        if not (stats1 and stats2):
            return {}
            
        return {
            'period1': {
                'mean': stats1[0],
                'min': stats1[1],
                'max': stats1[2],
                'count': stats1[3]
            },
            'period2': {
                'mean': stats2[0],
                'min': stats2[1],
                'max': stats2[2],
                'count': stats2[3]
            },
            'differences': {
                'mean_diff': stats2[0] - stats1[0],
                'mean_diff_percent': ((stats2[0] - stats1[0]) / stats1[0] * 100) if stats1[0] else 0,
                'range_diff': (stats2[2] - stats2[1]) - (stats1[2] - stats1[1])
            }
        }

    def export_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = 'csv',
        platforms: Optional[List[str]] = None
    ) -> str:
        """Export metrics data in various formats.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            format: Export format ('csv' or 'json')
            platforms: Optional list of platforms to include
            
        Returns:
            String containing the exported data
        """
        query = """
            SELECT timestamp, platform, metric_name, metric_value
            FROM metrics
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]
        
        if platforms:
            placeholders = ','.join('?' * len(platforms))
            query += f" AND platform IN ({placeholders})"
            params.extend(platforms)
        
        query += " ORDER BY timestamp, platform, metric_name"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            data = [dict(row) for row in cursor.fetchall()]
        
        if format == 'json':
            return json.dumps(data, default=str)
        else:  # csv
            output = StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return output.getvalue()

    def optimize_database(self):
        """Perform database maintenance and optimization."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Analyze tables for query optimization
            conn.execute("ANALYZE metrics")
            
            # Remove any fragmentation
            conn.execute("VACUUM")
            
            # Update table statistics
            conn.execute("ANALYZE sqlite_master")
            
            # Optimize indices
            conn.execute("REINDEX")

    @lru_cache(maxsize=1000)
    def get_cached_metrics(
        self,
        platform: str,
        metric_name: str,
        hours: int = 24
    ) -> List[Dict[str, Union[datetime, float]]]:
        """Get metrics with caching for improved performance."""
        cache_key = f"metrics:{platform}:{metric_name}:{hours}"
        
        # Try to get from cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, get from database
        data = self.get_trend_data(metric_name, platform, hours)
        
        # Store in cache for future use
        self.cache.set(cache_key, data, expire=3600)  # Cache for 1 hour
        
        return data
