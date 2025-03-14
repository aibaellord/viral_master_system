import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
from scipy.stats import beta
import logging
from datetime import datetime
import asyncio
from collections import defaultdict

@dataclass
class Decision:
    id: str
    context: Dict[str, Any]
    options: List[str]
    probabilities: List[float]
    chosen_option: str
    confidence: float
    timestamp: datetime
    
class DecisionEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_history: List[Decision] = []
        self.reward_history = defaultdict(list)
        self.contextual_data = defaultdict(dict)
        self.learning_rates = defaultdict(lambda: 0.1)
        
    async def make_decision(self, context: Dict[str, Any], options: List[str]) -> Decision:
        decision_scores = await asyncio.gather(
            self._calculate_bayesian_score(context, options),
            self._apply_reinforcement_learning(context, options),
            self._apply_fuzzy_logic(context, options),
            self._apply_game_theory(context, options)
        )
        
        combined_scores = np.mean(decision_scores, axis=0)
        probabilities = self._softmax(combined_scores)
        
        chosen_index = np.random.choice(len(options), p=probabilities)
        chosen_option = options[chosen_index]
        confidence = probabilities[chosen_index]
        
        decision = Decision(
            id=str(len(self.decision_history) + 1),
            context=context,
            options=options,
            probabilities=probabilities.tolist(),
            chosen_option=chosen_option,
            confidence=float(confidence),
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        return decision
        
    async def _calculate_bayesian_score(self, context: Dict[str, Any], options: List[str]) -> np.ndarray:
        scores = np.zeros(len(options))
        for i, option in enumerate(options):
            successes = sum(1 for r in self.reward_history[option] if r > 0.5)
            trials = len(self.reward_history[option]) + 1
            scores[i] = beta.rvs(successes + 1, trials - successes + 1)
        return scores
        
    async def _apply_reinforcement_learning(self, context: Dict[str, Any], options: List[str]) -> np.ndarray:
        scores = np.zeros(len(options))
        for i, option in enumerate(options):
            state_key = str(context)
            if state_key in self.contextual_data[option]:
                q_value = self.contextual_data[option][state_key]
                scores[i] = q_value
            else:
                scores[i] = random.random()  # Exploration
        return scores
        
    async def _apply_fuzzy_logic(self, context: Dict[str, Any], options: List[str]) -> np.ndarray:
        scores = np.zeros(len(options))
        for i, option in enumerate(options):
            membership_scores = []
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    membership = self._calculate_fuzzy_membership(value)
                    membership_scores.append(membership)
            scores[i] = np.mean(membership_scores) if membership_scores else 0.5
        return scores
        
    async def _apply_game_theory(self, context: Dict[str, Any], options: List[str]) -> np.ndarray:
        payoff_matrix = np.zeros((len(options), len(options)))
        for i, option1 in enumerate(options):
            for j, option2 in enumerate(options):
                payoff_matrix[i,j] = self._calculate_payoff(option1, option2)
        
        return np.mean(payoff_matrix, axis=1)
        
    def _calculate_fuzzy_membership(self, value: float) -> float:
        # Implement fuzzy membership function
        return 1 / (1 + np.exp(-value))
        
    def _calculate_payoff(self, option1: str, option2: str) -> float:
        # Implement game theory payoff calculation
        history1 = np.mean(self.reward_history[option1]) if self.reward_history[option1] else 0.5
        history2 = np.mean(self.reward_history[option2]) if self.reward_history[option2] else 0.5
        return (history1 + 1) / (history2 + 1)
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
        
    async def update_reward(self, decision_id: str, reward: float):
        try:
            decision = next(d for d in self.decision_history if d.id == decision_id)
            self.reward_history[decision.chosen_option].append(reward)
            
            # Update Q-values for reinforcement learning
            state_key = str(decision.context)
            current_q = self.contextual_data[decision.chosen_option].get(state_key, 0.0)
            learning_rate = self.learning_rates[decision.chosen_option]
            new_q = current_q + learning_rate * (reward - current_q)
            self.contextual_data[decision.chosen_option][state_key] = new_q
            
            # Adaptive learning rate
            self.learning_rates[decision.chosen_option] *= 0.995  # Decay learning rate
            
        except StopIteration:
            self.logger.error(f"Decision {decision_id} not found")
            
    def get_decision_metrics(self) -> Dict[str, float]:
        metrics = {
            "total_decisions": len(self.decision_history),
            "average_confidence": np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0,
            "option_distribution": {
                option: len([d for d in self.decision_history if d.chosen_option == option])
                for option in set(d.chosen_option for d in self.decision_history)
            }
        }
        return metrics

