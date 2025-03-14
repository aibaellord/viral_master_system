import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class ContentPerformanceOptimizer:
    """
    Advanced AI-driven content performance optimization system that analyzes, predicts, 
    and enhances content performance across multiple platforms.
    
    Features:
    - Content engagement prediction and analysis
    - A/B testing framework for content optimization
    - Platform-specific optimization strategies
    - Machine learning model management
    - Performance analytics and reporting
    """
    
    # Platform-specific configuration
    PLATFORM_CONFIGS = {
        'twitter': {
            'optimal_length_range': (80, 150),
            'hashtag_count_range': (1, 3),
            'engagement_metrics': ['likes', 'retweets', 'replies', 'impressions'],
            'primary_metric': 'retweets'
        },
        'instagram': {
            'optimal_length_range': (125, 300),
            'hashtag_count_range': (5, 15),
            'engagement_metrics': ['likes', 'comments', 'saves', 'shares', 'reach'],
            'primary_metric': 'likes'
        },
        'facebook': {
            'optimal_length_range': (100, 250),
            'hashtag_count_range': (0, 3),
            'engagement_metrics': ['reactions', 'comments', 'shares', 'reach'],
            'primary_metric': 'shares'
        },
        'linkedin': {
            'optimal_length_range': (200, 500),
            'hashtag_count_range': (3, 5),
            'engagement_metrics': ['reactions', 'comments', 'shares', 'impressions'],
            'primary_metric': 'shares'
        },
        'tiktok': {
            'optimal_length_range': (15, 60),  # video length in seconds
            'hashtag_count_range': (3, 7),
            'engagement_metrics': ['likes', 'comments', 'shares', 'views', 'watch_time'],
            'primary_metric': 'shares'
        },
        'youtube': {
            'optimal_length_range': (8, 15),  # video length in minutes
            'hashtag_count_range': (3, 5),
            'engagement_metrics': ['likes', 'comments', 'shares', 'views', 'watch_time'],
            'primary_metric': 'watch_time'
        }
    }
    
    def __init__(self, data_dir: str = "./data/performance", model_dir: str = "./models", 
                 log_level: int = logging.INFO):
        """
        Initialize the ContentPerformanceOptimizer with directories for data and model storage.
        
        Args:
            data_dir: Directory to store performance data
            model_dir: Directory to store trained ML models
            log_level: Logging level
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ContentPerformanceOptimizer")
        
        # Initialize models dictionary
        self.models = {}
        self.scalers = {}
        
        # A/B testing data
        self.ab_tests = {}
        
        # Performance data
        self.performance_data = defaultdict(list)
        
        # Load existing data and models if available
        self._load_performance_data()
        self._load_models()
        
        self.logger.info("ContentPerformanceOptimizer initialized")
    
    def predict_engagement(self, content: Dict[str, Any], platform: str) -> Dict[str, float]:
        """
        Predict engagement metrics for content on a specific platform using trained models.
        
        Args:
            content: Dictionary with content features
            platform: Platform to predict engagement for (twitter, instagram, etc.)
            
        Returns:
            Dictionary of predicted engagement metrics
        """
        if platform not in self.PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform}")
            
        # Extract features from content
        features = self._extract_content_features(content, platform)
        
        # Check if model exists for this platform
        if platform not in self.models:
            self.logger.warning(f"No model available for {platform}. Using baseline prediction.")
            return self._baseline_prediction(platform)
        
        # Scale features
        scaled_features = self.scalers[platform].transform([features])
        
        # Get predictions
        predictions = {}
        metrics = self.PLATFORM_CONFIGS[platform]['engagement_metrics']
        
        for metric in metrics:
            model_key = f"{platform}_{metric}"
            if model_key in self.models:
                try:
                    predictions[metric] = float(self.models[model_key].predict(scaled_features)[0])
                except Exception as e:
                    self.logger.error(f"Error predicting {metric} for {platform}: {str(e)}")
                    predictions[metric] = self._baseline_prediction(platform)[metric]
            else:
                predictions[metric] = self._baseline_prediction(platform)[metric]
        
        # Calculate overall engagement score
        predictions['engagement_score'] = self._calculate_engagement_score(predictions, platform)
        
        return predictions
    
    def optimize_content(self, content: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """
        Optimize content for better performance on a specific platform.
        
        Args:
            content: Dictionary with content data to optimize
            platform: Target platform
            
        Returns:
            Optimized content dictionary
        """
        if platform not in self.PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform}")
        
        # Deep copy content to avoid modifying original
        optimized = content.copy()
        
        # Apply platform-specific optimizations
        optimized = self._optimize_for_platform(optimized, platform)
        
        # Apply ML-based optimizations if model is available
        if platform in self.models:
            optimized = self._apply_ml_optimizations(optimized, platform)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(content, optimized, platform)
        optimized['optimization_recommendations'] = recommendations
        
        # Add performance prediction
        optimized['performance_prediction'] = self.predict_engagement(optimized, platform)
        
        return optimized
    
    def record_performance(self, content_id: str, platform: str, content: Dict[str, Any], 
                           metrics: Dict[str, float]) -> None:
        """
        Record actual performance metrics for content to use in model training.
        
        Args:
            content_id: Unique identifier for the content
            platform: Platform where content was published
            content: The content that was published
            metrics: Dictionary of performance metrics
        """
        if platform not in self.PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform}")
        
        # Extract features
        features = self._extract_content_features(content, platform)
        
        # Create record
        record = {
            'content_id': content_id,
            'platform': platform,
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'metrics': metrics,
            'content_summary': self._summarize_content(content)
        }
        
        # Add to performance data
        self.performance_data[platform].append(record)
        
        # Save data
        self._save_performance_data()
        
        self.logger.info(f"Recorded performance for content {content_id} on {platform}")
        
        # Check if we have enough data to retrain
        if len(self.performance_data[platform]) % 20 == 0:  # Retrain every 20 samples
            self.train_models(platforms=[platform])
    
    def create_ab_test(self, test_id: str, platform: str, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create an A/B test with multiple content variants.
        
        Args:
            test_id: Unique identifier for the test
            platform: Platform for the test
            variants: List of content variant dictionaries
            
        Returns:
            Dictionary with test details
        """
        if platform not in self.PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform}")
        
        if len(variants) < 2:
            raise ValueError("A/B test requires at least 2 variants")
        
        # Create test
        test = {
            'test_id': test_id,
            'platform': platform,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'variants': [],
            'results': None
        }
        
        # Add variants with predictions
        for i, variant in enumerate(variants):
            variant_id = f"{test_id}_variant_{i+1}"
            predictions = self.predict_engagement(variant, platform)
            
            test['variants'].append({
                'variant_id': variant_id,
                'content': variant,
                'predictions': predictions,
                'metrics': None
            })
        
        # Store test
        self.ab_tests[test_id] = test
        self._save_ab_tests()
        
        self.logger.info(f"Created A/B test {test_id} for {platform} with {len(variants)} variants")
        return test
    
    def record_ab_test_result(self, test_id: str, variant_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Record result for an A/B test variant.
        
        Args:
            test_id: Test identifier
            variant_id: Variant identifier
            metrics: Engagement metrics for the variant
            
        Returns:
            Updated test dictionary
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"Test ID not found: {test_id}")
        
        test = self.ab_tests[test_id]
        variant_found = False
        
        # Update variant metrics
        for variant in test['variants']:
            if variant['variant_id'] == variant_id:
                variant['metrics'] = metrics
                variant_found = True
                break
        
        if not variant_found:
            raise ValueError(f"Variant ID not found: {variant_id}")
        
        # Check if all variants have metrics
        all_metrics_received = all(v['metrics'] is not None for v in test['variants'])
        
        # If all metrics received, analyze results
        if all_metrics_received:
            test['status'] = 'completed'
            test['completed_at'] = datetime.now().isoformat()
            test['results'] = self._analyze_ab_test_results(test)
        
        # Save tests
        self._save_ab_tests()
        
        self.logger.info(f"Recorded result for variant {variant_id} in test {test_id}")
        return test
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get results for an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results dictionary
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"Test ID not found: {test_id}")
        
        test = self.ab_tests[test_id]
        
        if test['status'] != 'completed':
            self.logger.warning(f"Test {test_id} is not yet completed")
            
        return test
    
    def train_models(self, platforms: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Train prediction models for specified platforms using recorded performance data.
        
        Args:
            platforms: List of platforms to train models for. If None, train for all platforms
                      with sufficient data.
            
        Returns:
            Dictionary with training results per platform and metric
        """
        if platforms is None:
            platforms = [p for p in self.performance_data.keys() if len(self.performance_data[p]) >= 30]
        
        results = {}
        
        for platform in platforms:
            if platform not in self.PLATFORM_CONFIGS:
                self.logger.warning(f"Skipping unknown platform: {platform}")
                continue
                
            if len(self.performance_data.get(platform, [])) < 30:
                self.logger.warning(f"Not enough data to train model for {platform}. Need at least 30 samples.")
                continue
            
            platform_results = {}
            metrics = self.PLATFORM_CONFIGS[platform]['engagement_metrics']
            
            # Prepare data
            X, y_dict = self._prepare_training_data(platform)
            
            if len(X) == 0:
                self.logger.warning(f"No valid training data for {platform}")
                continue
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[platform] = scaler
            
            # Train a model for each metric
            for metric in metrics:
                if metric not in y_dict or len(y_dict[metric]) == 0:
                    self.logger.warning(f"No data for metric {metric} on platform {platform}")
                    continue
                
                y = y_dict[metric]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train models
                models = {
                    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
                }
                
                best_model = None
                best_score = -float('inf')
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.logger.info(f"{platform} - {metric} - {name}: MSE={mse:.4f}, R²={r2:.4f}")
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                
                if best_model is not None:
                    model_key = f"{platform}_{metric}"
                    self.models[model_key] = best_model
                    platform_results[metric] = {
                        'mse': float(mean_squared_error(y_test, best_model.predict(X_test))),
                        'r2': float(r2_score(y_test, best_model.predict(X_test)))
                    }
                    
                    # Save model
                    self._save

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content Performance Optimizer Module

This module provides advanced AI-driven optimization for content performance
across various platforms using machine learning models, engagement prediction,
A/B testing, and performance analytics.
"""

import json
import logging
import os
import pickle
import random
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enum representing different content types"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    STORY = "story"
    REEL = "reel"
    CAROUSEL = "carousel"


class Platform(Enum):
    """Enum representing different social media platforms"""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    PINTEREST = "pinterest"


class PerformanceMetric(Enum):
    """Enum representing different performance metrics"""
    ENGAGEMENT_RATE = "engagement_rate"
    REACH = "reach"
    IMPRESSIONS = "impressions"
    CLICKS = "clicks"
    SHARES = "shares"
    COMMENTS = "comments"
    LIKES = "likes"
    SAVES = "saves"
    WATCH_TIME = "watch_time"
    RETENTION_RATE = "retention_rate"


class ContentPerformanceOptimizer:
    """
    Advanced AI-driven content performance optimizer.
    
    This class provides methods for optimizing content performance through
    machine learning, A/B testing, trend analysis, and engagement prediction.
    """
    
    def __init__(self, 
                data_dir: str = "./data/performance", 
                model_dir: str = "./models",
                use_gpu: bool = False):
        """
        Initialize the ContentPerformanceOptimizer.
        
        Args:
            data_dir: Directory for storing performance data
            model_dir: Directory for storing trained models
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self.models = {}
        self.scalers = {}
        self.ab_tests = {}
        self.performance_data = {}
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load existing models if available
        self._load_models()
        
        # Load existing performance data
        self._load_performance_data()
        
        logger.info("ContentPerformanceOptimizer initialized successfully")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        for platform in Platform:
            platform_dir = self.data_dir / platform.value
            platform_dir.mkdir(exist_ok=True)
    
    def _load_models(self):
        """Load trained ML models if they exist"""
        if not self.model_dir.exists():
            logger.info("No models directory found. Will train new models as needed.")
            return
        
        model_files = list(self.model_dir.glob("*.pkl"))
        
        if not model_files:
            logger.info("No existing models found. Will train new models as needed.")
            return
        
        for model_file in model_files:
            try:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict) and 'model' in model_data and 'scaler' in model_data:
                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data['scaler']
                    logger.info(f"Loaded model: {model_name}")
                else:
                    logger.warning(f"Invalid model format in {model_file}")
                    
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {str(e)}")
    
    def _load_performance_data(self):
        """Load existing performance data"""
        performance_file = self.data_dir / "performance_data.json"
        
        if not performance_file.exists():
            logger.info("No performance data found. Starting with empty dataset.")
            return
        
        try:
            with open(performance_file, 'r') as f:
                self.performance_data = json.load(f)
            logger.info(f"Loaded performance data with {len(self.performance_data)} entries")
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            self.performance_data = {}
    
    def _save_performance_data(self):
        """Save performance data to disk"""
        performance_file = self.data_dir / "performance_data.json"
        
        try:
            with open(performance_file, 'w') as f:
                json.dump(self.performance_data, f)
            logger.info(f"Saved performance data with {len(self.performance_data)} entries")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
    
    def _save_model(self, model_name: str, model, scaler=None):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            model: The trained model object
            scaler: The scaler used for feature preprocessing
        """
        model_file = self.model_dir / f"{model_name}.pkl"
        
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved model: {model_name}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
    
    def _extract_content_features(self, content: Dict, content_type: ContentType) -> Dict:
        """
        Extract relevant features from content for machine learning models.
        
        Args:
            content: The content dictionary
            content_type: Type of the content
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'content_type': content_type.value,
            'length': 0,
            'has_hashtags': 0,
            'num_hashtags': 0,
            'has_emojis': 0,
            'num_emojis': 0,
            'has_mentions': 0,
            'num_mentions': 0,
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
        }
        
        # Extract text-specific features
        if content_type == ContentType.TEXT:
            if 'text' in content:
                text = content['text']
                features['length'] = len(text)
                features['has_hashtags'] = 1 if '#' in text else 0
                features['num_hashtags'] = text.count('#')
                features['has_emojis'] = 1 if any(c in text for c in ['😀', '😍', '👍', '🔥']) else 0
                features['num_emojis'] = sum(text.count(c) for c in ['😀', '😍', '👍', '🔥'])
                features['has_mentions'] = 1 if '@' in text else 0
                features['num_mentions'] = text.count('@')
                
                # Sentiment analysis (simplified)
                positive_words = ['amazing', 'awesome', 'great', 'excellent', 'love', 'best']
                negative_words = ['bad', 'terrible', 'worst', 'hate', 'awful', 'poor']
                
                text_lower = text.lower()
                features['sentiment_score'] = sum(1 for word in positive_words if word in text_lower) - \
                                             sum(1 for word in negative_words if word in text_lower)
        
        # Image-specific features
        elif content_type == ContentType.IMAGE or content_type == ContentType.CAROUSEL:
            if 'images' in content:
                features['num_images'] = len(content['images'])
                features['has_people'] = 1 if content.get('has_people', False) else 0
                features['image_brightness'] = content.get('brightness', 0.5)
                features['image_contrast'] = content.get('contrast', 0.5)
        
        # Video-specific features
        elif content_type in [ContentType.VIDEO, ContentType.REEL]:
            if 'duration' in content:
                features['length'] = content['duration']
                features['has_music'] = 1 if content.get('has_music', False) else 0
                features['has_captions'] = 1 if content.get('has_captions', False) else 0
        
        # Add custom features if provided
        if 'features' in content:
            for key, value in content['features'].items():
                if isinstance(value, (int, float, bool)):
                    features[key] = float(value)
        
        return features
    
    def _prepare_training_data(self, platform: Platform, content_type: ContentType = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for model training.
        
        Args:
            platform: Target platform
            content_type: Optional filter for content type
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        data = []
        targets = []
        
        for content_id, entry in self.performance_data.items():
            if entry['platform'] != platform.value:
                continue
                
            if content_type and entry['content_type'] != content_type.value:
                continue
            
            # Skip entries without performance metrics
            if 'metrics' not in entry or not entry['metrics']:
                continue
            
            features = entry['features']
            
            # Convert to proper format for ML
            features_dict = {k: float(v) if isinstance(v, (int, float, bool)) else v 
                           for k, v in features.items()}
            
            # Add categorical features as numerical
            for cat_feature in ['content_type', 'time_of_day', 'day_of_week']:
                if cat_feature in features_dict and isinstance(features_dict[cat_feature], str):
                    features_dict[cat_feature] = hash(features_dict[cat_feature]) % 10  # Simple hash to numeric
            
            data.append(features_dict)
            
            # Use engagement_rate as target by default, fallback to first available metric
            target_value = entry['metrics'].get(
                PerformanceMetric.ENGAGEMENT_RATE.value,
                list(entry['metrics'].values())[0] if entry['metrics'] else 0
            )
            targets.append(float(target_value))
        
        if not data:
            logger.warning(f"No training data available for {platform.value}" + 
                          (f" and {content_type.value}" if content_type else ""))
            return pd.DataFrame(), pd.DataFrame()
        
        # Create dataframes
        features_df = pd.DataFrame(data)
        targets_df = pd.DataFrame(targets, columns=['target'])
        
        return features_df, targets_df
    
    def train_model(self, 
                   platform: Platform, 
                   content_type: ContentType = None,
                   model_type: str = 'gradient_boosting') -> bool:
        """
        Train a machine learning model for performance prediction.
        
        Args:
            platform: Target platform
            content_type: Optional content type filter
            model_type: Type of model to train ('random_forest' or 'gradient_boosting')
            
        Returns:
            Boolean indicating training success
        """
        logger.info(f"Training model for {platform.value}" + 
                   (f" and {content_type.value}" if content_type else ""))
        
        # Prepare model name
        model_name = f"{platform.value}" + (f"_{content_type.value}" if content_type else "")
        
        # Get training data
        features_df, targets_df = self._prepare_training_data(platform, content_type)
        
        if features_df.empty or targets_df.empty:
            logger.warning(f"Insufficient data to train model for {model_name}")
            return False
        
        try:
            # Handle categorical features
            for col in features_df.select_dtypes(include=['object']).columns:
                features_df[col] = features_df[col].astype('category').cat.codes
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets_df, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select model type
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # default to gradient boosting
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train_scaled, y_train.values.ravel())
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model training completed for {model_name}:")
            logger.info(f"- Mean Squared Error: {mse:.4f}")
            logger.info(f"- R² Score: {r2:.4f}")
            
            # Save model
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self._save_model(model_name, model, scaler)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {model_name}: {str(e)}")
            return False
    
    def predict_performance(self, 
                           content: Dict, 
                           platform: Platform, 
                           content_type: ContentType) -> Dict[str, float]:
        """
        Predict performance metrics for content on a specific platform.
        

import os
import json
import numpy as np
import pandas as pd
import pickle
import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ContentPerformanceOptimizer:
    """
    Advanced AI-driven content performance optimization system.
    
    This system tracks content performance, performs A/B testing, analyzes
    engagement metrics, and applies machine learning to optimize future content
    for maximum viral potential.
    """
    
    def __init__(self, data_dir: str = "./data/performance/", 
                model_dir: str = "./models/", 
                min_samples_for_training: int = 50,
                feature_extraction_method: str = "tfidf",
                primary_model: str = "random_forest",
                confidence_threshold: float = 0.95,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the content performance optimizer.
        
        Args:
            data_dir: Directory for storing performance data
            model_dir: Directory for storing trained models
            min_samples_for_training: Minimum number of samples required before training models
            feature_extraction_method: Method to use for text feature extraction
            primary_model: Primary machine learning model to use
            confidence_threshold: Statistical significance threshold for A/B tests
            logger: Optional logger instance
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.min_samples_for_training = min_samples_for_training
        self.feature_extraction_method = feature_extraction_method
        self.primary_model = primary_model
        self.confidence_threshold = confidence_threshold
        
        # Initialize logger
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize data storage
        self._ensure_directories_exist()
        self.content_data = self._load_content_data()
        self.ab_tests = self._load_ab_test_data()
        
        # Initialize models
        self.engagement_model = None
        self.feature_extractor = None
        self.scaler = StandardScaler()
        
        # Load or initialize models
        self._initialize_models()
        
        self.logger.info("ContentPerformanceOptimizer initialized")
    
    def _ensure_directories_exist(self) -> None:
        """Ensure that necessary directories exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _load_content_data(self) -> pd.DataFrame:
        """Load content performance data from disk or initialize empty DataFrame."""
        data_path = os.path.join(self.data_dir, "content_performance.csv")
        if os.path.exists(data_path):
            try:
                return pd.read_csv(data_path)
            except Exception as e:
                self.logger.error(f"Error loading content data: {e}")
                return pd.DataFrame(columns=[
                    'content_id', 'platform', 'content_type', 'content_text', 
                    'publish_time', 'views', 'likes', 'shares', 'comments',
                    'clicks', 'watch_time', 'engagement_score'
                ])
        else:
            return pd.DataFrame(columns=[
                'content_id', 'platform', 'content_type', 'content_text', 
                'publish_time', 'views', 'likes', 'shares', 'comments',
                'clicks', 'watch_time', 'engagement_score'
            ])
    
    def _load_ab_test_data(self) -> Dict:
        """Load A/B test data from disk or initialize empty dictionary."""
        data_path = os.path.join(self.data_dir, "ab_tests.json")
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading A/B test data: {e}")
                return {"tests": [], "results": {}}
        else:
            return {"tests": [], "results": {}}
    
    def _initialize_models(self) -> None:
        """Initialize or load pre-trained models."""
        model_path = os.path.join(self.model_dir, f"{self.primary_model}_engagement.pkl")
        vectorizer_path = os.path.join(self.model_dir, f"{self.feature_extraction_method}_vectorizer.pkl")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
        
        # Try to load existing models
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.engagement_model = pickle.load(f)
                self.logger.info(f"Loaded engagement model from {model_path}")
            
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.feature_extractor = pickle.load(f)
                self.logger.info(f"Loaded feature extractor from {vectorizer_path}")
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Loaded feature scaler from {scaler_path}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self._initialize_new_models()
    
    def _initialize_new_models(self) -> None:
        """Initialize new models when pre-trained models aren't available."""
        # Initialize feature extractor
        if self.feature_extraction_method == "tfidf":
            self.feature_extractor = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3)
            )
        
        # Initialize engagement model
        if self.primary_model == "random_forest":
            self.engagement_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
        elif self.primary_model == "gradient_boosting":
            self.engagement_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
        self.logger.info("Initialized new models")
    
    def record_content_performance(self, 
                                  content_id: str,
                                  platform: str,
                                  content_type: str,
                                  content_text: str,
                                  metrics: Dict[str, float]) -> float:
        """
        Record content performance metrics and calculate engagement score.
        
        Args:
            content_id: Unique identifier for the content
            platform: Platform where content was published
            content_type: Type of content (text, image, video, etc.)
            content_text: Text content or description
            metrics: Dictionary of performance metrics (views, likes, shares, etc.)
            
        Returns:
            Calculated engagement score
        """
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(platform, metrics)
        
        # Create a new record
        new_record = {
            'content_id': content_id,
            'platform': platform,
            'content_type': content_type,
            'content_text': content_text,
            'publish_time': datetime.datetime.now().isoformat(),
            'views': metrics.get('views', 0),
            'likes': metrics.get('likes', 0),
            'shares': metrics.get('shares', 0),
            'comments': metrics.get('comments', 0),
            'clicks': metrics.get('clicks', 0),
            'watch_time': metrics.get('watch_time', 0),
            'engagement_score': engagement_score
        }
        
        # Add to dataframe
        self.content_data = pd.concat([self.content_data, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save updated data
        self._save_content_data()
        
        # Retrain model if we have enough data
        if len(self.content_data) % 10 == 0 and len(self.content_data) >= self.min_samples_for_training:
            self.train_engagement_model()
        
        return engagement_score
    
    def record_ab_test_result(self, 
                             test_id: str,
                             variant_id: str,
                             content: str,
                             metrics: Dict[str, float],
                             platform: str) -> Dict[str, Any]:
        """
        Record the result of an A/B test variant.
        
        Args:
            test_id: Unique identifier for the A/B test
            variant_id: Identifier for the specific variant
            content: Content used in this variant
            metrics: Performance metrics for this variant
            platform: Platform where the test was conducted
            
        Returns:
            Dictionary with test statistics including engagement score
        """
        # Calculate engagement score based on platform and metrics
        engagement_score = self._calculate_engagement_score(platform, metrics)
        
        # Extract content features
        features = self._extract_content_features(content)
        
        # Create result record
        result = {
            'test_id': test_id,
            'variant_id': variant_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'platform': platform,
            'content': content,
            'metrics': metrics,
            'engagement_score': engagement_score,
            'features': features
        }
        
        # Initialize test if it doesn't exist
        if test_id not in self.ab_tests['results']:
            self.ab_tests['results'][test_id] = []
            
            # Add to test list if not already there
            if test_id not in self.ab_tests['tests']:
                self.ab_tests['tests'].append(test_id)
        
        # Add result to test
        self.ab_tests['results'][test_id].append(result)
        
        # Save AB test data
        self._save_ab_test_data()
        
        # Calculate test statistics if we have multiple variants
        test_stats = {}
        if len(self.ab_tests['results'][test_id]) > 1:
            test_stats = self._analyze_ab_test(test_id)
        
        return {
            'engagement_score': engagement_score,
            'test_stats': test_stats,
            'percentile': self._calculate_engagement_percentile(engagement_score, platform)
        }
    
    def _calculate_engagement_score(self, platform: str, metrics: Dict[str, float]) -> float:
        """
        Calculate engagement score based on platform and metrics.
        
        Different platforms weight metrics differently:
        - Instagram: Emphasizes likes and comments
        - Twitter: Emphasizes retweets and replies
        - Facebook: Balances reactions, comments, and shares
        - TikTok: Emphasizes watch time and shares
        - YouTube: Emphasizes watch time and subscribers gained
        - LinkedIn: Emphasizes clicks and comments
        
        Args:
            platform: Social media platform
            metrics: Dictionary of performance metrics
            
        Returns:
            Calculated engagement score (0-100)
        """
        # Base weights for common metrics
        weights = {
            'views': 0.1,
            'likes': 1.0,
            'shares': 5.0,
            'comments': 2.0,
            'clicks': 3.0,
            'watch_time': 0.5,  # Per second
        }
        
        # Platform-specific adjustments
        if platform.lower() == 'instagram':
            weights['likes'] = 1.5
            weights['comments'] = 3.0
            weights['saves'] = 5.0
        elif platform.lower() == 'twitter':
            weights['retweets'] = 5.0  # Retweets are shares
            weights['replies'] = 2.0   # Replies are comments
            weights['likes'] = 0.8
        elif platform.lower() == 'tiktok':
            weights['watch_time'] = 2.0
            weights['shares'] = 8.0
            weights['completions'] = 4.0
        elif platform.lower() == 'youtube':
            weights['watch_time'] = 1.5
            weights['subscribers_gained'] = 10.0
            weights['likes'] = 0.5
        elif platform.lower() == 'linkedin':
            weights['clicks'] = 5.0
            weights['comments'] = 3.0
            weights['impressions'] = 0.05
        
        # Calculate engagement score
        score = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                score += value * weights[metric]
            elif metric == 'retweets' and platform.lower() == 'twitter':
                score += value * weights['shares']
            elif metric == 'replies' and platform.lower() == 'twitter':
                score += value * weights['comments']
        
        # Normalize score (0-100)
        # This is a simplified approach; real normalization would require platform benchmarks
        norm_factor = max(1, metrics.get('views', 0)) / 100
        normalized_score = min(100, score / norm_factor)
        
        return round(normalized_score, 2)
    
    def _calculate_engagement_percentile(self, score: float, platform: str) -> float:
        """
        Calculate the percentile of an engagement score compared to historical data.
        
        Args:
            score: Engagement score to evaluate
            platform: Platform to compare against
            
        Returns:
            Percentile value (0-100)
        """
        if len(self.content_data) == 0:
            return 50.0  # Default to median if no data
            
        platform_data = self.content_data[self.content_data['platform'] == platform]
        if len(platform_data) == 0:
            return 50.0  # Default to median if no data for platform
        
        # Calculate percentile
        percentile = stats.percentileofscore(platform_data['engagement_score'], score)
        return round(percentile, 2)
    
    def _extract_content_features(self, content: str) -> Dict[str, float]:
        """
        Extract features from content text for analysis.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary of extracted features
        """
        # Basic text metrics
        features = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len(content.split('.')),
            'avg_word_length': sum(len(word) for word in content.split()) / max(1, len(content.split())),
            'question_count': content.count('?'),
            'exclamation_count': content.count('!'),
            'hashtag_count': content.count('#'),
            'mention_count': content.count('@'),
            'url_count

import os
import json
import numpy as np
import pandas as pd
import logging
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# For A/B testing
from scipy import stats

# Type aliases for improved readability
ContentData = Dict[str, Any]
MetricsData = Dict[str, float]
PerformanceData = Dict[str, Union[str, float, Dict]]

class ContentPerformanceOptimizer:
    """
    Advanced AI-driven content performance optimization system that monitors, analyzes,
    and optimizes content performance across different platforms.
    
    Features:
    - Content performance monitoring
    - AI-driven content optimization
    - Content engagement analytics
    - Performance prediction
    - Content A/B testing
    - Cross-platform analytics
    """
    
    def __init__(self, 
                 data_dir: str = "./data/performance", 
                 models_dir: str = "./models",
                 log_level: int = logging.INFO):
        """
        Initialize the ContentPerformanceOptimizer with directories for data and models.
        
        Args:
            data_dir: Directory to store performance data
            models_dir: Directory to store trained models
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger("ContentPerformanceOptimizer")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Data storage paths
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/ab_tests", exist_ok=True)
        
        # Performance data by platform
        self.performance_data = defaultdict(list)
        self._load_performance_data()
        
        # Engagement metrics tracking
        self.engagement_metrics = {
            "views": [],
            "likes": [],
            "comments": [],
            "shares": [],
            "clicks": [],
            "conversions": [],
            "retention": []
        }
        
        # Machine learning models for different platforms
        self.models = {}
        self.scalers = {}
        
        # A/B test tracking
        self.ab_tests = {}
        self._load_ab_tests()
        
        # Content optimization suggestions cache
        self.optimization_cache = {}
        
        # Performance thresholds for different content types
        self.performance_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "average": 0.4,
            "below_average": 0.2
        }
        
        # Platform-specific feature importances (initial values)
        self.platform_features = {
            "twitter": ["brevity", "hashtags", "timeliness", "emotion"],
            "instagram": ["visual_appeal", "storytelling", "hashtags", "aesthetic"],
            "facebook": ["engagement", "emotion", "shareability", "relevance"],
            "linkedin": ["professionalism", "value", "insights", "authority"],
            "tiktok": ["hook_strength", "trend_alignment", "audio_quality", "pacing"],
            "youtube": ["thumbnail", "title", "first_30s", "value_delivery"]
        }
        
        self.logger.info("ContentPerformanceOptimizer initialized successfully")
    
    def record_content_performance(self, 
                                  content_id: str, 
                                  platform: str, 
                                  content_data: ContentData,
                                  performance_metrics: MetricsData) -> None:
        """
        Record the performance of a piece of content on a specific platform.
        
        Args:
            content_id: Unique identifier for the content
            platform: Platform where content was published
            content_data: Content details including text, structure, etc.
            performance_metrics: Key metrics like views, likes, comments, etc.
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(platform, performance_metrics)
        
        # Extract content features
        content_features = self._extract_content_features(content_data, platform)
        
        # Create performance record
        performance_record = {
            "content_id": content_id,
            "platform": platform,
            "timestamp": timestamp,
            "content_data": content_data,
            "metrics": performance_metrics,
            "engagement_score": engagement_score,
            "features": content_features
        }
        
        # Add to in-memory store
        self.performance_data[platform].append(performance_record)
        
        # Save to disk
        self._save_performance_data(platform, performance_record)
        
        # Update models if we have enough data
        if len(self.performance_data[platform]) % 10 == 0:  # Every 10 records
            self._update_prediction_model(platform)
        
        self.logger.info(f"Recorded performance for content {content_id} on {platform} with score {engagement_score:.2f}")
        
        # Clear optimization cache for this platform
        if platform in self.optimization_cache:
            del self.optimization_cache[platform]
    
    def get_content_analytics(self, 
                             platform: Optional[str] = None, 
                             time_period: Optional[int] = None,
                             content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive analytics for content performance.
        
        Args:
            platform: Optional filter by platform
            time_period: Optional time period in days (None for all-time)
            content_type: Optional filter by content type
            
        Returns:
            Dictionary containing analytics data
        """
        # Filter data based on parameters
        filtered_data = self._filter_performance_data(platform, time_period, content_type)
        
        if not filtered_data:
            return {"error": "No data available for the specified filters"}
        
        # Calculate key metrics
        total_content = len(filtered_data)
        platforms = set(item["platform"] for item in filtered_data)
        
        # Engagement metrics
        avg_engagement = sum(item["engagement_score"] for item in filtered_data) / total_content
        
        # Performance distribution
        performance_distribution = {
            "excellent": len([i for i in filtered_data if i["engagement_score"] >= self.performance_thresholds["excellent"]]),
            "good": len([i for i in filtered_data if self.performance_thresholds["good"] <= i["engagement_score"] < self.performance_thresholds["excellent"]]),
            "average": len([i for i in filtered_data if self.performance_thresholds["average"] <= i["engagement_score"] < self.performance_thresholds["good"]]),
            "below_average": len([i for i in filtered_data if self.performance_thresholds["below_average"] <= i["engagement_score"] < self.performance_thresholds["average"]]),
            "poor": len([i for i in filtered_data if i["engagement_score"] < self.performance_thresholds["below_average"]])
        }
        
        # Top performing content
        top_content = sorted(filtered_data, key=lambda x: x["engagement_score"], reverse=True)[:5]
        top_content_ids = [item["content_id"] for item in top_content]
        
        # Time-based trends
        if time_period:
            # Group by day and calculate average engagement
            time_trend = self._calculate_time_trends(filtered_data)
        else:
            time_trend = None
        
        # Feature importance if we have trained models
        feature_importance = {}
        for platform in platforms:
            if platform in self.models:
                feature_importance[platform] = self._get_feature_importance(platform)
        
        return {
            "total_content": total_content,
            "platforms": list(platforms),
            "average_engagement": avg_engagement,
            "performance_distribution": performance_distribution,
            "top_content_ids": top_content_ids,
            "time_trend": time_trend,
            "feature_importance": feature_importance
        }
    
    def predict_performance(self, 
                           content_data: ContentData, 
                           platform: str) -> Dict[str, Any]:
        """
        Predict the performance of content before publishing.
        
        Args:
            content_data: Content to predict performance for
            platform: Target platform
            
        Returns:
            Dictionary with predicted metrics and confidence
        """
        # Extract features from content
        features = self._extract_content_features(content_data, platform)
        
        # Convert features to array for prediction
        feature_array = np.array([list(features.values())])
        
        # Check if we have a model for this platform
        if platform not in self.models:
            # If no model, estimate based on rules
            return self._rule_based_prediction(content_data, platform)
        
        # Scale features
        scaled_features = self.scalers[platform].transform(feature_array)
        
        # Predict using model
        predicted_score = self.models[platform].predict(scaled_features)[0]
        
        # Calculate confidence (based on model's feature importances and data variance)
        confidence = self._calculate_prediction_confidence(platform, features)
        
        # Generate expected metrics based on the score
        expected_metrics = self._generate_expected_metrics(predicted_score, platform)
        
        # Generate optimization suggestions
        optimization_suggestions = self.suggest_optimizations(content_data, platform)
        
        return {
            "predicted_score": float(predicted_score),
            "confidence": confidence,
            "expected_metrics": expected_metrics,
            "optimization_suggestions": optimization_suggestions,
            "performance_category": self._get_performance_category(predicted_score)
        }
    
    def suggest_optimizations(self, 
                             content_data: ContentData, 
                             platform: str) -> List[Dict[str, Any]]:
        """
        Suggest optimizations to improve content performance.
        
        Args:
            content_data: Content to optimize
            platform: Target platform
            
        Returns:
            List of optimization suggestions with expected impact
        """
        # Check cache first
        cache_key = f"{hash(json.dumps(content_data, sort_keys=True))}_{platform}"
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Extract current content features
        current_features = self._extract_content_features(content_data, platform)
        
        suggestions = []
        
        # If we have a model, use it for optimization
        if platform in self.models:
            # Find feature importances
            feature_importances = self._get_feature_importance(platform)
            
            # Focus on top features for optimization
            top_features = sorted(
                feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # Focus on top 3 features
            
            for feature_name, importance in top_features:
                # Get current feature value
                current_value = current_features.get(feature_name, 0)
                
                # Get suggestions based on feature
                suggestion = self._generate_feature_optimization(
                    feature_name, current_value, content_data, platform
                )
                
                if suggestion:
                    suggestion["expected_impact"] = importance * 0.1  # Scale importance to impact
                    suggestions.append(suggestion)
        else:
            # Rule-based suggestions if no model available
            suggestions = self._generate_rule_based_suggestions(content_data, platform)
        
        # Add platform-specific suggestions
        platform_suggestions = self._generate_platform_specific_suggestions(content_data, platform)
        suggestions.extend(platform_suggestions)
        
        # Cache results
        self.optimization_cache[cache_key] = suggestions
        
        return suggestions
    
    def create_ab_test(self, 
                      test_name: str, 
                      original_content: ContentData,
                      variant_content: ContentData,
                      platform: str,
                      test_hypothesis: str) -> str:
        """
        Create an A/B test to compare two content variants.
        
        Args:
            test_name: Name of the test
            original_content: Original content (A)
            variant_content: Variant content to test (B)
            platform: Platform for the test
            test_hypothesis: Hypothesis being tested
            
        Returns:
            Unique test ID
        """
        test_id = f"abtest_{int(datetime.datetime.now().timestamp())}_{test_name.replace(' ', '_')}"
        
        # Create test record
        test_record = {
            "test_id": test_id,
            "test_name": test_name,
            "platform": platform,
            "hypothesis": test_hypothesis,
            "start_time": datetime.datetime.now().isoformat(),
            "status": "running",
            "variants": {
                "A": {
                    "content": original_content,
                    "metrics": {},
                    "sample_size": 0
                },
                "B": {
                    "content": variant_content,
                    "metrics": {},
                    "sample_size": 0
                }
            },
            "results": None
        }
        
        # Store test
        self.ab_tests[test_id] = test_record
        self._save_ab_test(test_id)
        
        self.logger.info(f"Created A/B test '{test_name}' with ID {test_id}")
        
        return test_id
    
    def record_ab_test_result(self, 
                             test_id: str, 
                             variant: str, 
                             metrics: MetricsData) -> Dict[str, Any]:
        """
        Record results for an A/B test variant.
        
        Args:
            test_id: Test identifier
            variant: Variant identifier ('A' or 'B')
            metrics: Performance metrics
            
        Returns:
            Current test status
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test with ID {test_id} not found")
        
        if variant not in ("A", "B"):
            raise ValueError("Variant must be either 'A' or 'B'")
        
        test = self.ab_tests[test_id]
        
        # Update sample size
        test["variants"][variant]["sample_size"] += 1
        
        # Aggregate metrics
        current_metrics = test["variants"][variant]["metrics"]
        
        for key, value in metrics.items():
            if key in current_metrics:
                # Compute running average
                n = test["variants"][variant]["sample_size"]
                current_metrics[key] = (current_metrics[key] * (n-1) + value) / n
            else:
                current_metrics[key] = value
        
        # Calculate engagement score
        platform = test["platform"]
        engagement_score = self._

