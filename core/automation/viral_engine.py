from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.ml.predictor import MLPredictor
from core.metrics.types import ViralMetrics, NetworkMetrics
from core.platforms.base_platform_client import BasePlatformClient
from core.optimization.optimizer import ContentOptimizer

@dataclass
class ViralLoopConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    prediction_window: int = 24  # hours
    min_confidence: float = 0.85
    max_viral_coefficient: float = 5.0
    network_sample_size: int = 10000

class ViralPredictionNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, 4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        attn_out, _ = self.attention(out, out, out)
        return self.fc(attn_out.mean(dim=1))

class ViralLoopEngine:
    def __init__(
        self,
        config: ViralLoopConfig,
        predictor: MLPredictor,
        platform_clients: List[BasePlatformClient],
        optimizer: ContentOptimizer
    ):
        self.config = config
        self.predictor = predictor
        self.platform_clients = platform_clients
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize viral prediction network
        self.viral_net = ViralPredictionNet(input_size=len(ViralMetrics.__members__))
        self.viral_net.to(self.device)
        
        # Initialize network analysis graph
        self.network_graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)

    async def amplify_network_effects(self, content_id: str) -> float:
        """Amplifies network effects for given content through multi-platform optimization."""
        metrics = await self._gather_cross_platform_metrics(content_id)
        network_graph = self._build_influence_graph(metrics)
        
        # Identify high-impact nodes and optimize their engagement
        influential_nodes = self._identify_influential_nodes(network_graph)
        amplification_factor = await self._optimize_node_engagement(influential_nodes)
        
        return amplification_factor

    async def optimize_viral_coefficient(self, content_id: str) -> Tuple[float, Dict]:
        """Optimizes viral coefficient through real-time monitoring and adjustment."""
        initial_coefficient = await self._calculate_viral_coefficient(content_id)
        spread_patterns = await self._analyze_spread_patterns(content_id)
        
        optimization_params = {
            'share_probability': spread_patterns['share_rate'],
            'conversion_rate': spread_patterns['conversion_rate'],
            'audience_reach': spread_patterns['reach_factor']
        }
        
        # Use neural net to predict optimal parameters
        optimized_params = await self._optimize_viral_parameters(optimization_params)
        new_coefficient = await self._apply_coefficient_optimization(content_id, optimized_params)
        
        return new_coefficient, optimized_params

    async def detect_share_triggers(self, content_id: str) -> List[Dict]:
        """Detects and analyzes sharing triggers using ML-based pattern recognition."""
        share_events = await self._gather_share_events(content_id)
        
        # Process share events through neural network
        event_tensors = torch.tensor([event['features'] for event in share_events])
        with torch.no_grad():
            trigger_scores = self.viral_net(event_tensors.to(self.device))
        
        triggers = []
        for event, score in zip(share_events, trigger_scores):
            if score > self.config.min_confidence:
                triggers.append({
                    'type': event['type'],
                    'context': event['context'],
                    'confidence': float(score),
                    'impact_score': float(score * event['reach'])
                })
        
        return sorted(triggers, key=lambda x: x['impact_score'], reverse=True)

    async def optimize_user_action_loop(self, content_id: str) -> Dict:
        """Optimizes user action loops for maximum viral spread."""
        current_loops = await self._analyze_user_actions(content_id)
        
        # Build and analyze action graph
        action_graph = nx.DiGraph()
        for action in current_loops:
            action_graph.add_weighted_edges_from(action['transitions'])
        
        # Optimize action paths
        optimal_paths = nx.floyd_warshall(action_graph)
        
        return await self._optimize_action_paths(optimal_paths)

    async def analyze_spread_patterns(self, content_id: str) -> Dict:
        """Analyzes viral spread patterns using graph algorithms and ML predictions."""
        metrics = await self._gather_cross_platform_metrics(content_id)
        
        # Build spread graph
        spread_graph = self._build_spread_graph(metrics)
        
        # Calculate spread metrics
        spread_velocity = self._calculate_spread_velocity(spread_graph)
        growth_rate = self._calculate_growth_rate(metrics)
        viral_paths = self._identify_viral_paths(spread_graph)
        
        return {
            'velocity': spread_velocity,
            'growth_rate': growth_rate,
            'viral_paths': viral_paths,
            'bottlenecks': self._identify_spread_bottlenecks(spread_graph)
        }

    async def synchronize_cross_platform(self, content_id: str) -> Dict:
        """Synchronizes viral optimization across all platforms."""
        platform_metrics = {}
        futures = []
        
        # Gather metrics from all platforms asynchronously
        for client in self.platform_clients:
            futures.append(client.get_content_metrics(content_id))
        
        results = await asyncio.gather(*futures)
        
        # Normalize and analyze cross-platform performance
        for client, metrics in zip(self.platform_clients, results):
            platform_metrics[client.platform_name] = self._normalize_metrics(metrics)
        
        # Optimize cross-platform strategy
        strategy = await self._optimize_cross_platform_strategy(platform_metrics)
        
        return {
            'platform_metrics': platform_metrics,
            'optimization_strategy': strategy,
            'sync_recommendations': self._generate_sync_recommendations(strategy)
        }

    def _build_influence_graph(self, metrics: Dict) -> nx.DiGraph:
        """Builds a graph representation of influence network."""
        G = nx.DiGraph()
        
        for user_id, user_metrics in metrics['user_interactions'].items():
            G.add_node(user_id, 
                      influence_score=user_metrics['influence_score'],
                      engagement_rate=user_metrics['engagement_rate'])
            
            for follower in user_metrics['followers']:
                G.add_edge(user_id, follower, 
                          weight=user_metrics['follower_engagement'][follower])
        
        return G

    def _identify_influential_nodes(self, G: nx.DiGraph) -> List[str]:
        """Identifies most influential nodes using PageRank and other metrics."""
        pagerank_scores = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Combine metrics for overall influence score
        influence_scores = {}
        for node in G.nodes():
            influence_scores[node] = (
                pagerank_scores[node] * 0.4 +
                betweenness[node] * 0.3 +
                G.nodes[node]['influence_score'] * 0.3
            )
        
        return sorted(influence_scores.keys(), 
                     key=lambda x: influence_scores[x], 
                     reverse=True)[:self.config.network_sample_size]

    async def _optimize_viral_parameters(self, params: Dict) -> Dict:
        """Optimizes viral parameters using neural network predictions."""
        param_tensor = torch.tensor([list(params.values())], dtype=torch.float32)
        
        with torch.no_grad():
            optimized_values = self.viral_net(param_tensor.to(self.device))
        
        optimized_params = {}
        for param, value in zip(params.keys(), optimized_values[0]):
            optimized_params[param] = float(value)
        
        return optimized_params

    def _calculate_spread_velocity(self, G: nx.DiGraph) -> float:
        """Calculates the velocity of content spread through the network."""
        timestamps = nx.get_node_attributes(G, 'timestamp')
        if not timestamps:
            return 0.0
        
        # Calculate time-based spread metrics
        start_time = min(timestamps.values())
        end_time = max(timestamps.values())
        time_diff = (end_time - start_time).total_seconds() / 3600  # hours
        
        if time_diff == 0:
            return 0.0
        
        # Calculate spread metrics
        nodes_reached = len(G.nodes())
        spread_velocity = nodes_reached / time_diff
        
        return spread_velocity

    async def _gather_cross_platform_metrics(self, content_id: str) -> Dict:
        """Gathers and aggregates metrics from all platforms."""
        metrics = {}
        futures = []
        
        for client in self.platform_clients:
            futures.append(client.get_content_metrics(content_id))
        
        results = await asyncio.gather(*futures)
        
        for client, result in zip(self.platform_clients, results):
            metrics[client.platform_name] = result
        
        return self._aggregate_cross_platform_metrics(metrics)

    def _aggregate_cross_platform_metrics(self, metrics: Dict) -> Dict:
        """Aggregates metrics across platforms into unified metrics."""
        aggregated = {
            'total_reach': 0,
            'engagement_rate': 0,
            'viral_coefficient': 0,
            'share_rate': 0,
            'platform_breakdown': {},
            'user_interactions': {}
        }
        
        total_users = 0
        for platform, data in metrics.items():
            total_users += data['unique_users']
            aggregated['total_reach'] += data['reach']
            aggregated['platform_breakdown'][platform] = {
                'contribution_rate': data['reach'] / max(1, aggregated['total_reach']),
                'engagement_rate': data['engagement_rate'],
                'viral_coefficient': data['viral_coefficient']
            }
            
            # Merge user interactions
            for user_id, interactions in data['user_interactions'].items():
                if user_id not in aggregated['user_interactions']:
                    aggregated['user_interactions'][user_id] = interactions
                else:
                    # Merge interaction data
                    existing = aggregated['user_interactions'][user_id]
                    existing['influence_score'] = max(
                        existing['influence_score'],
                        interactions['influence_score']
                    )
                    existing['followers'].update(interactions['followers'])
                    existing['follower_engagement'].update(
                        interactions['follower_engagement']
                    )
        
        # Calculate overall metrics
        if total_users > 0:
            aggregated['engagement_rate'] = sum(
                p['engagement_rate'] * p['contribution_rate']
                for p in aggregated['platform_breakdown'].values()
            )
            aggregated['viral_coefficient'] = sum(
                p['viral_coefficient'] * p['contribution_rate']
                for p in aggregated['platform_breakdown'].values()
            )
        
        return aggregated

