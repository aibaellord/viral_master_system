"""
FeedbackLoop: Autonomous feedback and self-evolution engine.
- Logs all content/campaign results (engagement, shares, income, etc.)
- Triggers retraining/fine-tuning of local models (LLM, RL, image, etc.)
- Interfaces with RL/AutoML engine for continuous improvement.
"""
import logging
from typing import Dict, Any, List, Optional
import datetime
import json

class FeedbackLoop:
    def __init__(self, retrain_callback=None, log_file: str = "feedback_log.jsonl"):
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.retrain_callback = retrain_callback
        self.records: List[Dict[str, Any]] = []

    def log_result(self, content_id: str, metrics: Dict[str, Any], meta: Optional[Dict[str, Any]] = None):
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "content_id": content_id,
            "metrics": metrics,
            "meta": meta or {}
        }
        self.records.append(record)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log feedback: {e}")

    def should_retrain(self, threshold: float = 0.7, window: int = 100) -> bool:
        # Check if average performance drops below threshold
        if len(self.records) < window:
            return False
        scores = [r["metrics"].get("composite_score", 0.0) for r in self.records[-window:]]
        avg_score = sum(scores) / len(scores)
        return avg_score < threshold

    def trigger_retrain(self):
        self.logger.info("Triggering retraining/fine-tuning pipeline...")
        if self.retrain_callback:
            self.retrain_callback()

    def feedback_step(self, content_id: str, metrics: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, threshold: float = 0.7, window: int = 100):
        self.log_result(content_id, metrics, meta)
        if self.should_retrain(threshold, window):
            self.trigger_retrain()
