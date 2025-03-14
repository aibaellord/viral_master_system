import os
import json
import numpy as np
import pandas as pd
import pickle
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional, Union

class PerformanceOptimizer:
    """
    AI-Powered Performance Optimization System for content generation.
    
    This class handles:
    - Performance data collection and analysis
    - Machine learning model for content optimization
    - A/B testing with statistical significance analysis
    - Feedback loop system for continuous improvement
    - Performance metrics tracking and visualization
    
    The system uses machine learning to analyze successful content and 
    continuously improve content generation based on real-world performance.
    """
    
    def __init__(self, model_dir: str = "./models", data_dir: str = "./data"):
        """
        Initialize the Performance Optimizer.
        
        Args:
            model_dir: Directory to store trained models
            data_dir: Directory to store performance data
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.ab_test_results = {}
        self.performance_data = []
        self.model = None
        self.scaler = StandardScaler()
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "ab_tests"), exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PerformanceOptimizer')
        
        # Load existing data and models if available
        self._load_performance_data()
        self._load_model()
        self._load_ab_test_results()
    
    def record_content_performance(self, content_id: str, content_features: Dict[str, Any], 
                                  performance_metrics: Dict[str, float]) -> None:
        """
        Record performance data for a piece of content.
        
        Args:
            content_id: Unique identifier for the content
            content_features: Features of the content (e.g., emotional_triggers, topic, platform)
            performance_metrics: Performance metrics (e.g., engagement, shares, likes)
        """
        timestamp = datetime.datetime.now().isoformat()
        
        performance_record = {
            "content_id": content_id,
            "timestamp": timestamp,
            "features": content_features,
            "metrics": performance_metrics
        }
        
        self.performance_data.append(performance_record)
        self._save_performance_data()
        self.logger.info(f"Recorded performance for content ID {content_id}")
    
    def train_optimization_model(self) -> Dict[str, float]:
        """
        Train a machine learning model on the collected performance data.
        
        Returns:
            Dictionary containing training metrics like R-squared, MSE, etc.
        """
        if len(self.performance_data) < 10:
            self.logger.warning("Not enough data to train model (minimum 10 records needed)")
            return {"status": "insufficient_data", "records_count": len(self.performance_data)}
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X.shape[0] == 0:
            self.logger.warning("No valid features extracted from performance data")
            return {"status": "invalid_data", "records_count": len(self.performance_data)}
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Save model
        self._save_model()
        
        metrics = {
            "status": "success",
            "records_count": len(self.performance_data),
            "training_r2": train_score,
            "testing_r2": test_score,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Model trained successfully. Test R-squared: {test_score:.4f}")
        return metrics
    
    def optimize_content_parameters(self, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize content parameters based on trained model.
        
        Args:
            content_features: Initial content features to optimize
            
        Returns:
            Dictionary with optimized content parameters
        """
        if self.model is None:
            self.logger.warning("No trained model available. Using original parameters.")
            return content_features
        
        # Extract and prepare features
        optimized_features = content_features.copy()
        
        # If we have enough data, perform optimization
        if len(self.performance_data) >= 50:
            # This is a simplified optimization approach
            # In a full implementation, we would explore the parameter space more thoroughly
            
            # Generate variations of the content features
            variations = self._generate_parameter_variations(content_features)
            
            # Predict performance for each variation
            best_score = -1
            best_variation = content_features
            
            for variation in variations:
                # Convert to feature vector
                features_vector = self._features_to_vector(variation)
                if features_vector is None:
                    continue
                
                # Scale features
                features_scaled = self.scaler.transform([features_vector])
                
                # Predict performance
                predicted_score = self.model.predict(features_scaled)[0]
                
                if predicted_score > best_score:
                    best_score = predicted_score
                    best_variation = variation
            
            optimized_features = best_variation
            self.logger.info(f"Content parameters optimized. Predicted improvement: {best_score:.2f}")
        
        return optimized_features
    
    def setup_ab_test(self, test_id: str, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Set up an A/B test with multiple content variants.
        
        Args:
            test_id: Unique identifier for the test
            variants: List of content variant configurations
            
        Returns:
            Dictionary with test details
        """
        timestamp = datetime.datetime.now().isoformat()
        
        ab_test = {
            "test_id": test_id,
            "created_at": timestamp,
            "status": "active",
            "variants": variants,
            "results": {variant.get("variant_id", f"variant_{i}"): {"impressions": 0, "engagements": 0, "conversions": 0}
                      for i, variant in enumerate(variants)}
        }
        
        self.ab_test_results[test_id] = ab_test
        self._save_ab_test_results()
        
        self.logger.info(f"Created A/B test with ID {test_id} and {len(variants)} variants")
        return ab_test
    
    def record_ab_test_result(self, test_id: str, variant_id: str, 
                             metrics: Dict[str, Union[int, float]]) -> Dict[str, Any]:
        """
        Record the result for a specific variant in an A/B test.
        
        Args:
            test_id: ID of the A/B test
            variant_id: ID of the variant
            metrics: Performance metrics for this instance (impressions, engagements, conversions, etc.)
            
        Returns:
            Updated test results
        """
        if test_id not in self.ab_test_results:
            self.logger.error(f"A/B test with ID {test_id} not found")
            return {"error": f"Test ID {test_id} not found"}
        
        # Get the test data
        test_data = self.ab_test_results[test_id]
        
        # Check if variant exists
        if variant_id not in test_data["results"]:
            self.logger.error(f"Variant {variant_id} not found in test {test_id}")
            return {"error": f"Variant {variant_id} not found in test {test_id}"}
        
        # Update metrics
        for metric, value in metrics.items():
            if metric in test_data["results"][variant_id]:
                test_data["results"][variant_id][metric] += value
            else:
                test_data["results"][variant_id][metric] = value
        
        # Record timestamp of last update
        test_data["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save updated results
        self._save_ab_test_results()
        
        self.logger.info(f"Recorded results for variant {variant_id} in test {test_id}")
        
        # Check if we have enough data for significance testing
        if self._check_test_significance_threshold(test_id):
            self.analyze_ab_test(test_id)
        
        return test_data
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the results of an A/B test.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dictionary with test results and analysis
        """
        if test_id not in self.ab_test_results:
            self.logger.error(f"A/B test with ID {test_id} not found")
            return {"error": f"Test ID {test_id} not found"}
        
        test_data = self.ab_test_results[test_id]
        
        # If we have enough data, add statistical analysis
        if self._check_test_significance_threshold(test_id):
            analysis = self.analyze_ab_test(test_id)
            test_data["analysis"] = analysis
        else:
            test_data["analysis"] = {
                "status": "insufficient_data",
                "message": "Not enough data collected for meaningful analysis"
            }
        
        return test_data
    
    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """
        Perform statistical analysis on A/B test results.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dictionary with statistical analysis
        """
        if test_id not in self.ab_test_results:
            self.logger.error(f"A/B test with ID {test_id} not found")
            return {"error": f"Test ID {test_id} not found"}
        
        test_data = self.ab_test_results[test_id]
        variants = list(test_data["results"].keys())
        
        if len(variants) < 2:
            return {"error": "At least two variants are required for analysis"}
        
        # Calculate conversion rates
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {},
            "comparisons": []
        }
        
        # Calculate key metrics for each variant
        for variant_id, results in test_data["results"].items():
            impressions = results.get("impressions", 0)
            engagements = results.get("engagements", 0)
            conversions = results.get("conversions", 0)
            
            # Avoid division by zero
            engagement_rate = engagements / impressions if impressions > 0 else 0
            conversion_rate = conversions / impressions if impressions > 0 else 0
            
            analysis["metrics"][variant_id] = {
                "impressions": impressions,
                "engagements": engagements,
                "conversions": conversions,
                "engagement_rate": engagement_rate,
                "conversion_rate": conversion_rate
            }
        
        # Perform pairwise statistical comparisons
        for i, variant1 in enumerate(variants):
            for variant2 in variants[i+1:]:
                comparison = self._compare_variants(
                    test_data["results"][variant1],
                    test_data["results"][variant2],
                    variant1,
                    variant2
                )
                analysis["comparisons"].append(comparison)
        
        # Determine winner
        if analysis["comparisons"]:
            # Find variant with highest conversion rate
            conversion_rates = [(v, m["conversion_rate"]) for v, m in analysis["metrics"].items()]
            winner, best_rate = max(conversion_rates, key=lambda x: x[1])
            
            # Check if winner is statistically significant
            has_significant_winner = any(
                comp["is_significant"] and comp["better_variant"] == winner
                for comp in analysis["comparisons"]
            )
            
            if has_significant_winner:
                analysis["winner"] = {
                    "variant_id": winner,
                    "conversion_rate": best_rate,
                    "is_significant": True
                }
            else:
                analysis["winner"] = {
                    "status": "no_significant_winner",
                    "message": "No statistically significant winner determined yet"
                }
        
        # Update test data with analysis
        test_data["analysis"] = analysis
        self._save_ab_test_results()
        
        return analysis
    
    def generate_performance_report(self, days: int = 30, 
                                   export_format: str = "json") -> str:
        """
        Generate a performance report for the specified time period.
        
        Args:
            days: Number of days to include in the report
            export_format: Format for the report ('json', 'csv', 'html')
            
        Returns:
            Path to the exported report file
        """
        # Filter data for the specified time period
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_date_str = cutoff_date.isoformat()
        
        recent_data = [
            record for record in self.performance_data
            if record["timestamp"] > cutoff_date_str
        ]
        
        if not recent_data:
            self.logger.warning(f"No performance data found for the last {days} days")
            return ""
        
        # Create report directory if it doesn't exist
        report_dir = os.path.join(self.data_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.{export_format}"
        file_path = os.path.join(report_dir, filename)
        
        # Create and export report
        if export_format == "json":
            with open(file_path, "w") as f:
                json.dump({
                    "report_type": "performance_summary",
                    "period_days": days,
                    "generated_at": datetime.datetime.now().isoformat(),
                    

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import datetime
import logging
import random
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PerformanceOptimizer")

class PerformanceOptimizer:
    """
    Advanced performance optimization system for viral content generation.
    This class provides tools for collecting performance data, analyzing trends,
    training machine learning models, conducting A/B testing, and implementing
    feedback loops to continuously improve content performance.
    """
    
    def __init__(self, model_path: str = './models/performance_model.pkl'):
        """
        Initialize the performance optimizer.
        
        Args:
            model_path: Path to save/load the machine learning model
        """
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.scaler = StandardScaler()
        self.model = None
        self.performance_data = []
        self.ab_test_campaigns = {}
        self.feature_importance = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Load existing model if available
        if os.path.exists(self.model_path):
            try:
                self._load_model()
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Will train a new model when data is available.")
        else:
            logger.info("No existing model found. Will train a new model when data is available.")
    
    def collect_performance_data(self, content_data: Dict[str, Any], performance_metrics: Dict[str, float]) -> None:
        """
        Collect performance data for a piece of content.
        
        Args:
            content_data: Dictionary containing content features (platform, topic, structure, etc.)
            performance_metrics: Dictionary containing performance metrics (engagement, shares, likes, etc.)
        """
        # Combine content data and performance metrics
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "content_features": content_data,
            "performance_metrics": performance_metrics
        }
        
        self.performance_data.append(record)
        logger.info(f"Collected performance data for content: {content_data.get('content_id', 'unknown')}")
        
        # Save data to file
        self._save_performance_data()
    
    def analyze_performance_trends(self, metric: str = 'engagement_rate', 
                                 group_by: str = 'platform', 
                                 time_period: str = 'week') -> Dict[str, Any]:
        """
        Analyze performance trends across different dimensions.
        
        Args:
            metric: The performance metric to analyze
            group_by: The dimension to group by (platform, topic, emotion, etc.)
            time_period: Time period for aggregation (day, week, month)
            
        Returns:
            Dict containing trend analysis results
        """
        if not self.performance_data:
            logger.warning("No performance data available for trend analysis")
            return {"error": "No performance data available"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(record["timestamp"]),
                group_by: record["content_features"].get(group_by, "unknown"),
                metric: record["performance_metrics"].get(metric, 0)
            }
            for record in self.performance_data
        ])
        
        # Group by time period and the specified dimension
        if time_period == 'day':
            df['time_group'] = df['timestamp'].dt.date
        elif time_period == 'week':
            df['time_group'] = df['timestamp'].dt.isocalendar().week
        elif time_period == 'month':
            df['time_group'] = df['timestamp'].dt.month
        
        # Aggregate and compute statistics
        grouped = df.groupby(['time_group', group_by])[metric].agg(['mean', 'std', 'count'])
        
        # Detect trends
        pivot = grouped.reset_index().pivot(index='time_group', columns=group_by, values='mean')
        
        # Calculate growth rates
        growth_rates = {}
        for column in pivot.columns:
            if not pivot[column].isna().all() and len(pivot[column].dropna()) > 1:
                first_valid = pivot[column].first_valid_index()
                last_valid = pivot[column].last_valid_index()
                if first_valid is not None and last_valid is not None:
                    first_value = pivot[column][first_valid]
                    last_value = pivot[column][last_valid]
                    if first_value > 0:
                        growth_rate = ((last_value - first_value) / first_value) * 100
                        growth_rates[column] = growth_rate
        
        # Best performing categories
        if not pivot.empty:
            latest_data = pivot.iloc[-1].dropna()
            if not latest_data.empty:
                best_performers = latest_data.nlargest(3).index.tolist()
            else:
                best_performers = []
        else:
            best_performers = []
        
        return {
            "metric": metric,
            "group_by": group_by,
            "time_period": time_period,
            "trends": pivot.to_dict() if not pivot.empty else {},
            "growth_rates": growth_rates,
            "best_performers": best_performers
        }
    
    def train_optimization_model(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train a machine learning model to predict content performance.
        
        Args:
            force_retrain: Force retraining even if a model already exists
            
        Returns:
            Dict containing training results and model metrics
        """
        if not self.performance_data:
            logger.warning("No performance data available for model training")
            return {"error": "No performance data available for training"}
        
        if self.model is not None and not force_retrain:
            logger.info("Model already exists. Use force_retrain=True to retrain.")
            return {"status": "Model already trained", "action": "none"}
        
        # Prepare data for training
        features, targets = self._prepare_training_data()
        
        if len(features) < 10:
            logger.warning("Insufficient data for training (less than 10 samples)")
            return {"error": "Insufficient data for training", "samples_needed": 10 - len(features)}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save feature importance
        feature_names = self._get_feature_names()
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Save model
        self._save_model()
        
        return {
            "status": "Model trained successfully",
            "metrics": {
                "mse": mse,
                "r2": r2,
                "samples": len(features)
            },
            "feature_importance": dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        }
    
    def optimize_content(self, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize content features for better performance.
        
        Args:
            content_features: Dictionary of content features to optimize
            
        Returns:
            Dict containing optimized content features and predicted improvement
        """
        if self.model is None:
            logger.warning("No trained model available for optimization")
            return {"error": "No trained model available", "action": "train_model_first"}
        
        # Prepare features for prediction
        features_df = pd.DataFrame([content_features])
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            logger.warning("No numeric features found for optimization")
            return {"error": "No numeric features found for optimization"}
        
        # Create variations of the content
        variations = self._generate_content_variations(numeric_features)
        
        # Predict performance for each variation
        best_variation = None
        best_score = float('-inf')
        original_score = None
        
        for variation in variations:
            # Get prediction
            variation_scaled = self.scaler.transform([variation.values])
            score = self.model.predict(variation_scaled)[0]
            
            # Check if this is the original
            if np.array_equal(variation.values, numeric_features.values[0]):
                original_score = score
            
            # Track best variation
            if score > best_score:
                best_score = score
                best_variation = variation
        
        # Combine optimized numeric features with non-numeric ones
        optimized_features = content_features.copy()
        for col in best_variation.index:
            optimized_features[col] = best_variation[col]
        
        # Calculate improvement
        improvement = ((best_score - original_score) / original_score * 100) if original_score else 0
        
        return {
            "original_features": content_features,
            "optimized_features": optimized_features,
            "original_predicted_score": original_score,
            "optimized_predicted_score": best_score,
            "predicted_improvement": improvement
        }
    
    def setup_ab_test(self, 
                     test_name: str, 
                     variations: List[Dict[str, Any]], 
                     traffic_split: List[float] = None,
                     metrics: List[str] = None) -> Dict[str, Any]:
        """
        Set up an A/B test campaign.
        
        Args:
            test_name: Name of the A/B test campaign
            variations: List of content variations to test
            traffic_split: Percentage of traffic to allocate to each variation (must sum to 1.0)
            metrics: List of metrics to track for the test
            
        Returns:
            Dict containing A/B test configuration
        """
        if test_name in self.ab_test_campaigns:
            logger.warning(f"A/B test '{test_name}' already exists")
            return {"error": f"A/B test '{test_name}' already exists"}
        
        # Validate variations
        if not variations or len(variations) < 2:
            logger.error("At least 2 variations are required for A/B testing")
            return {"error": "At least 2 variations are required for A/B testing"}
        
        # Set default traffic split if not provided
        if traffic_split is None:
            traffic_split = [1.0 / len(variations)] * len(variations)
        
        # Validate traffic split
        if len(traffic_split) != len(variations):
            logger.error("Traffic split must match the number of variations")
            return {"error": "Traffic split must match the number of variations"}
        
        if abs(sum(traffic_split) - 1.0) > 0.001:
            logger.error("Traffic split must sum to 1.0")
            return {"error": "Traffic split must sum to 1.0"}
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ["engagement_rate", "click_through_rate", "conversion_rate"]
        
        # Create A/B test campaign
        ab_test = {
            "name": test_name,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "active",
            "variations": variations,
            "variation_names": [f"Variation {chr(65 + i)}" for i in range(len(variations))],
            "traffic_split": traffic_split,
            "metrics": metrics,
            "results": {metric: {f"Variation {chr(65 + i)}": [] for i in range(len(variations))} for metric in metrics},
            "impressions": {f"Variation {chr(65 + i)}": 0 for i in range(len(variations))},
            "statistical_significance": {metric: False for metric in metrics}
        }
        
        self.ab_test_campaigns[test_name] = ab_test
        logger.info(f"A/B test '{test_name}' created with {len(variations)} variations")
        
        return {"status": "success", "test": ab_test}
    
    def get_ab_test_variation(self, test_name: str) -> Dict[str, Any]:
        """
        Get a variation for a user based on the A/B test configuration.
        
        Args:
            test_name: Name of the A/B test campaign
            
        Returns:
            Dict containing the selected variation
        """
        if test_name not in self.ab_test_campaigns:
            logger.warning(f"A/B test '{test_name}' not found")
            return {"error": f"A/B test '{test_name}' not found"}
        
        ab_test = self.ab_test_campaigns[test_name]
        
        if ab_test["status"] != "active":
            logger.warning(f"A/B test '{test_name}' is not active")
            return {"error": f"A/B test '{test_name}' is not active"}
        
        # Select variation based on traffic split
        rand = random.random()
        cumulative = 0
        selected_variation = 0
        
        for i, split in enumerate(ab_test["traffic_split"]):
            cumulative += split
            if rand <= cumulative:
                selected_variation = i
                break
        
        # Increment impression count
        variation_name = f"Variation {chr(65 + selected_variation)}"
        ab_test["impressions"][variation_name] += 1
        
        return {
            "test_name": test_name,
            "variation_name": variation_name,
            "variation_index": selected_variation,
            "content": ab_test["variations"][selected_variation]
        }
    
    def record_ab_test_result(self, 
                             test_name: str, 
                             variation_name: str, 
                             metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Record the result of a specific variation for an A/B test.
        
        Args:
            test

