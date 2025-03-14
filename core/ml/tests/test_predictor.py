import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from core.ml.predictor import (
    PredictionConfig,
    TimeSeriesLSTM,
    ContentAnalyzer,
    ViralPredictor,
)
from core.metrics.types import MetricsSnapshot, ContentMetrics, AudienceMetrics
from core.platforms.base_platform_client import PlatformMetrics

@pytest.fixture
def prediction_config():
    return PredictionConfig(
        time_series_window=24,
        forecast_horizon=12,
        min_samples=10,
        confidence_interval=0.95,
        batch_size=8,
        learning_rate=0.001,
        max_epochs=10
    )

@pytest.fixture
def time_series_model():
    return TimeSeriesLSTM(
        input_size=10,
        hidden_size=32,
        num_layers=2
    )

@pytest.fixture
def content_analyzer():
    return ContentAnalyzer()

@pytest.fixture
def viral_predictor(prediction_config):
    return ViralPredictor(prediction_config)

@pytest.fixture
def sample_metrics():
    def create_metric(hours_ago: int) -> MetricsSnapshot:
        return MetricsSnapshot(
            timestamp=datetime.now() - timedelta(hours=hours_ago),
            views=1000 + hours_ago * 100,
            likes=500 + hours_ago * 50,
            shares=100 + hours_ago * 10,
            comments=50 + hours_ago * 5,
            engagement_rate=0.05 + hours_ago * 0.001,
            growth_rate=0.02 + hours_ago * 0.001,
            audience_size=10000 + hours_ago * 1000
        )
    
    return [create_metric(i) for i in range(48)]

def test_time_series_lstm():
    model = TimeSeriesLSTM(input_size=5, hidden_size=32, num_layers=2)
    batch_size = 4
    seq_length = 10
    
    x = torch.randn(batch_size, seq_length, 5)
    output = model(x)
    
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()

def test_content_analyzer(content_analyzer):
    content = "This is a test content piece that should be analyzed for sentiment and other features."
    
    features = content_analyzer.analyze_content(content)
    
    assert 'sentiment' in features
    assert 'length' in features
    assert 'word_count' in features
    assert 'avg_word_length' in features
    
    assert 0 <= features['sentiment'] <= 1
    assert features['length'] == len(content)
    assert features['word_count'] == len(content.split())

def test_viral_predictor_metric_prediction(viral_predictor, sample_metrics):
    predictions = viral_predictor.predict_metrics(sample_metrics)
    
    assert 'predictions' in predictions
    assert 'confidence_intervals' in predictions
    assert len(predictions['predictions']) > 0
    assert len(predictions['confidence_intervals']) == 2

def test_viral_score_calculation(viral_predictor):
    content = "Test content for viral score calculation"
    metrics = ContentMetrics(
        engagement_rate=0.05,
        share_rate=0.02,
        growth_rate=0.01
    )
    
    score = viral_predictor.calculate_viral_score(content, metrics)
    
    assert 0 <= score <= 1

def test_content_optimization(viral_predictor):
    content = "This is a test content piece that needs optimization suggestions"
    audience = AudienceMetrics(
        size=10000,
        growth_rate=0.02,
        engagement_rate=0.05
    )
    
    suggestions = viral_predictor.optimize_content(content, audience)
    
    assert 'sentiment_adjustment' in suggestions
    assert 'length_adjustment' in suggestions
    assert 'predicted_viral_score' in suggestions
    assert 'optimization_confidence' in suggestions

def test_ab_testing(viral_predictor):
    variants = [
        "First test variant for A/B testing",
        "Second test variant with different content"
    ]
    
    metrics = [
        PlatformMetrics(engagement_rate=0.05, share_rate=0.02, growth_rate=0.01),
        PlatformMetrics(engagement_rate=0.06, share_rate=0.03, growth_rate=0.02)
    ]
    
    results = viral_predictor.run_ab_test(variants, metrics)
    
    assert len(results) == len(variants)
    for variant in variants:
        assert 'viral_score' in results[variant]
        assert 'engagement_prediction' in results[variant]
        assert 'confidence' in results[variant]
        assert 'recommendation' in results[variant]

def test_trend_detection(viral_predictor, sample_metrics):
    trends = viral_predictor.detect_trends(sample_metrics)
    
    assert len(trends) > 0
    for metric_name, trend_data in trends.items():
        assert len(trend_data) > 0
        for trend in trend_data:
            assert 'timestamp' in trend
            assert 'value' in trend
            assert 'trend' in trend
            assert 'strength' in trend
            assert 'volatility' in trend

def test_confidence_intervals(viral_predictor):
    predictions = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [0.9, 1.9, 2.9]])
    
    lower_bound, upper_bound = viral_predictor._calculate_confidence_intervals(predictions)
    
    assert lower_bound.shape == predictions[0].shape
    assert upper_bound.shape == predictions[0].shape
    assert (lower_bound <= upper_bound).all()

@pytest.mark.integration
def test_full_prediction_pipeline(viral_predictor, sample_metrics):
    # Test the full pipeline from metrics to predictions and optimizations
    content = "This is a test content piece for the full prediction pipeline"
    
    # Predict metrics
    metric_predictions = viral_predictor.predict_metrics(sample_metrics)
    assert 'predictions' in metric_predictions
    
    # Calculate viral score
    metrics = ContentMetrics(
        engagement_rate=0.05

