"""
Metrics for evaluating agent performance.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import difflib


@dataclass
class TaskMetrics:
    """
    Container for task performance metrics.
    """
    episode_id: int
    accuracy: float
    correct_answers: int
    total_questions: int
    memory_used: int
    memory_capacity: int
    write_count: int
    memory_efficiency: float
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)


def similarity_score(predicted: str, expected: str) -> float:
    """
    Calculate similarity between predicted and expected answers.
    
    Uses string similarity ratio from difflib.
    Extracts key words from verbose answers (e.g., "golden" from "The token is golden").
    Range: 0.0 (completely different) to 1.0 (identical)
    
    Args:
        predicted: Predicted answer text (may be verbose or partial)
        expected: Expected answer text (usually a single word or short phrase)
        
    Returns:
        Similarity score (0-1)
    """
    predicted_lower = predicted.lower().strip()
    expected_lower = expected.lower().strip()
    
    if not predicted_lower or not expected_lower:
        return 0.0
    
    # Direct match on full strings
    if predicted_lower == expected_lower:
        return 1.0
    
    # Check if expected appears anywhere in predicted (substring match)
    if expected_lower in predicted_lower:
        return 1.0
    
    # Check if predicted appears anywhere in expected  
    if predicted_lower in expected_lower:
        return 0.9
    
    # Check if words match
    predicted_words = set(predicted_lower.split())
    expected_words = set(expected_lower.split())
    
    # Remove common filler words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'on', 'at', 'of', 'by', 'it', 'that', 'this'}
    predicted_words = predicted_words - stop_words
    expected_words = expected_words - stop_words
    
    # If any content word matches, it's a good match
    if predicted_words & expected_words:  # intersection
        return 0.95
    
    # Fall back to sequence matching on full strings
    ratio = difflib.SequenceMatcher(None, predicted_lower, expected_lower).ratio()
    
    # Boost score slightly if they're similar enough
    if ratio > 0.5:
        return ratio
    
    return 0.0


def exact_match_score(predicted: str, expected: str, threshold: float = 0.95) -> float:
    """
    Check if prediction matches expected answer.
    
    Args:
        predicted: Predicted answer
        expected: Expected answer
        threshold: Similarity threshold for match (default: 0.95)
        
    Returns:
        1.0 if match, 0.0 otherwise
    """
    sim = similarity_score(predicted, expected)
    return 1.0 if sim >= threshold else 0.0


class MetricsAggregator:
    """
    Aggregates metrics across multiple episodes.
    """
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.episodes_metrics: List[TaskMetrics] = []
    
    def add_episode(self, metrics: Dict) -> TaskMetrics:
        """
        Add metrics from one episode.
        
        Args:
            metrics: Dictionary of episode metrics
            
        Returns:
            TaskMetrics object
        """
        memory_efficiency = metrics['memory_used'] / max(metrics['memory_capacity'], 1)
        
        task_metrics = TaskMetrics(
            episode_id=metrics['episode_id'],
            accuracy=metrics['accuracy'],
            correct_answers=metrics['correct_answers'],
            total_questions=metrics['total_questions'],
            memory_used=metrics['memory_used'],
            memory_capacity=metrics['memory_capacity'],
            write_count=metrics['write_count'],
            memory_efficiency=memory_efficiency
        )
        
        self.episodes_metrics.append(task_metrics)
        return task_metrics
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics across all episodes.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.episodes_metrics:
            return {}
        
        accuracies = [m.accuracy for m in self.episodes_metrics]
        memory_used = [m.memory_used for m in self.episodes_metrics]
        write_counts = [m.write_count for m in self.episodes_metrics]
        memory_eff = [m.memory_efficiency for m in self.episodes_metrics]
        
        return {
            'num_episodes': len(self.episodes_metrics),
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'mean_memory_used': sum(memory_used) / len(memory_used),
            'mean_write_count': sum(write_counts) / len(write_counts),
            'mean_memory_efficiency': sum(memory_eff) / len(memory_eff),
            'total_correct': sum(m.correct_answers for m in self.episodes_metrics),
            'total_questions': sum(m.total_questions for m in self.episodes_metrics),
            'episodes': [m.to_dict() for m in self.episodes_metrics]
        }
    
    def get_learning_curve(self) -> List[float]:
        """
        Get learning curve as accuracy per episode.
        
        Returns:
            List of accuracies in episode order
        """
        return [m.accuracy for m in self.episodes_metrics]
    
    def get_memory_curve(self) -> List[float]:
        """
        Get memory usage curve.
        
        Returns:
            List of memory usage rates per episode
        """
        return [m.memory_efficiency for m in self.episodes_metrics]
    
    def clear(self):
        """Clear all accumulated metrics."""
        self.episodes_metrics.clear()


def evaluate_episode(agent_metrics: Dict, environment_ground_truth: Dict = None) -> Dict:
    """
    Evaluate a single episode comprehensively.
    
    Args:
        agent_metrics: Metrics from agent's episode run
        environment_ground_truth: Ground truth information about the episode
        
    Returns:
        Comprehensive evaluation dictionary
    """
    evaluation = {
        'accuracy': agent_metrics['accuracy'],
        'questions_correct': agent_metrics['correct_answers'],
        'questions_total': agent_metrics['total_questions'],
        'memory_utilization': agent_metrics['memory_stats']['utilization'],
        'memory_writes': agent_metrics['memory_stats']['write_count'],
        'question_details': agent_metrics['question_results']
    }
    
    # Analyze question results
    correct_questions = [r for r in agent_metrics['question_results'] if r['correct']]
    evaluation['correct_question_ids'] = [r['question_id'] for r in correct_questions]
    
    # Analyze retrieval effectiveness
    hit_rates = []
    for result in agent_metrics['question_results']:
        required_ids = set(result['required_facts'])
        retrieved_ids = set(result['retrieved_fact_ids'])
        hit_rate = len(required_ids & retrieved_ids) / max(len(required_ids), 1)
        hit_rates.append(hit_rate)
    
    if hit_rates:
        evaluation['mean_retrieval_hit_rate'] = sum(hit_rates) / len(hit_rates)
    
    return evaluation
