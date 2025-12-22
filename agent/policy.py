"""
Memory selection policies for the agent.
"""
import random
import numpy as np
from typing import List, Tuple
from env.streaming_qa_env import Fact
from agent.features import FeatureExtractor


class MemorySelectionPolicy:
    """
    Learned contextual bandit policy for memory selection.
    
    Uses logistic regression on fact features to decide whether to store.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize the memory selection policy.
        
        Args:
            learning_rate: Learning rate for policy updates
        """
        self.feature_extractor = FeatureExtractor()
        self.learning_rate = learning_rate
        
        # Initialize weights for linear policy
        feature_dim = self.feature_extractor.feature_dimension()
        self.weights = np.zeros(feature_dim)
        self.bias = 0.0
        
        # Tracking for learning
        self.episode_count = 0
        self.write_decisions = []
        self.write_rewards = []
    
    def decide(self, fact: Fact) -> bool:
        """
        Decide whether to write a fact to memory.
        
        Args:
            fact: The fact to make a decision about
            
        Returns:
            True if fact should be stored, False otherwise
        """
        features = np.array(self.feature_extractor.extract_features(fact))
        
        # Compute logits
        logits = np.dot(self.weights, features) + self.bias
        
        # Sigmoid for probability
        prob = 1.0 / (1.0 + np.exp(-logits))
        
        # Binary decision with threshold
        # Use >= 0.5 so initial random state (prob=0.5) defaults to storing facts
        return bool(prob >= 0.5)
    
    def get_decision_probability(self, fact: Fact) -> float:
        """
        Get the probability of storing a fact.
        
        Args:
            fact: The fact to evaluate
            
        Returns:
            Probability between 0 and 1
        """
        features = np.array(self.feature_extractor.extract_features(fact))
        logits = np.dot(self.weights, features) + self.bias
        prob = 1.0 / (1.0 + np.exp(-logits))
        return prob
    
    def update(self, fact: Fact, decision: bool, reward: float):
        """
        Update policy based on decision outcome.
        
        Args:
            fact: The fact that was written or skipped
            decision: Whether the fact was stored
            reward: Reward signal (1.0 if decision helped answer correctly, 0.0 otherwise)
        """
        features = np.array(self.feature_extractor.extract_features(fact))
        
        # Simple gradient update
        # Positive reward: increase likelihood of decision
        # Negative reward: decrease likelihood of decision
        gradient = features * (reward - 0.5)
        
        self.weights += self.learning_rate * gradient
        self.bias += self.learning_rate * (reward - 0.5)
    
    def update_episode(self, facts_and_decisions: List[Tuple[Fact, bool]], rewards: List[float]):
        """
        Update policy based on episode outcomes.
        
        Args:
            facts_and_decisions: List of (fact, decision) tuples
            rewards: List of rewards for each decision
        """
        for (fact, decision), reward in zip(facts_and_decisions, rewards):
            self.update(fact, decision, reward)
        
        self.episode_count += 1


class RandomPolicy:
    """Policy that makes random memory decisions."""
    
    def __init__(self, write_probability: float = 0.5):
        """
        Initialize random policy.
        
        Args:
            write_probability: Probability of storing any fact
        """
        self.write_probability = write_probability
    
    def decide(self, fact: Fact) -> bool:
        """
        Randomly decide whether to write a fact.
        
        Args:
            fact: The fact to decide about
            
        Returns:
            True with probability write_probability
        """
        return random.random() < self.write_probability
    
    def get_decision_probability(self, fact: Fact) -> float:
        """
        Get decision probability (constant for random policy).
        
        Args:
            fact: The fact to evaluate
            
        Returns:
            The write probability
        """
        return self.write_probability


class HeuristicPolicy:
    """
    Heuristic-based memory selection policy.
    
    Makes decisions based on manually defined rules.
    """
    
    def __init__(self):
        """Initialize heuristic policy."""
        self.feature_extractor = FeatureExtractor()
    
    def decide(self, fact: Fact) -> bool:
        """
        Decide based on heuristic rules.
        
        Args:
            fact: The fact to decide about
            
        Returns:
            True if fact passes heuristic checks
        """
        # Rule 1: Always skip distractors
        if fact.fact_type == 'distractor':
            return False
        
        # Rule 2: Always store identifiers (contain important info)
        if fact.fact_type == 'identifier':
            return True
        
        # Rule 3: Store attributes and events
        if fact.fact_type in ['attribute', 'event']:
            return True
        
        return False
    
    def get_decision_probability(self, fact: Fact) -> float:
        """
        Get decision probability (0.0 or 1.0 for deterministic policy).
        
        Args:
            fact: The fact to evaluate
            
        Returns:
            1.0 if would store, 0.0 otherwise
        """
        return 1.0 if self.decide(fact) else 0.0


class NoMemoryPolicy:
    """
    Baseline policy that stores nothing.
    """
    
    def decide(self, fact: Fact) -> bool:
        """
        Never store facts.
        
        Args:
            fact: The fact to decide about
            
        Returns:
            Always False
        """
        return False
    
    def get_decision_probability(self, fact: Fact) -> float:
        """
        Get decision probability (always 0).
        
        Args:
            fact: The fact to evaluate
            
        Returns:
            Always 0.0
        """
        return 0.0
