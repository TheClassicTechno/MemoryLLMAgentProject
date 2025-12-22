"""
Baseline implementations for comparison.
"""
from typing import Dict, List
from env.streaming_qa_env import Episode
from memory.memory_store import MemoryStore
from memory.retrieval import KeywordRetrieval
from agent.policy import RandomPolicy, HeuristicPolicy, NoMemoryPolicy


class BaselineAgent:
    """
    Baseline agent using different memory selection policies.
    """
    
    def __init__(self, policy, memory_capacity: int = 5, retrieval_budget: int = 3):
        """
        Initialize baseline agent.
        
        Args:
            policy: The policy to use for memory decisions
            memory_capacity: Maximum facts to store
            retrieval_budget: Maximum facts to retrieve
        """
        self.policy = policy
        self.memory_store = MemoryStore(capacity=memory_capacity)
        self.retrieval = KeywordRetrieval(top_k=retrieval_budget)
    
    def process_episode(self, episode: Episode) -> Dict:
        """
        Process an episode with the baseline policy.
        
        Args:
            episode: The episode to process
            
        Returns:
            Dictionary with results
        """
        self.memory_store.clear()
        
        # Phase 1: Store facts
        for fact in episode.facts:
            decision = self.policy.decide(fact)
            if decision:
                self.memory_store.write(fact)
        
        # Phase 2: Answer questions
        correct_count = 0
        question_results = []
        
        for question in episode.questions:
            stored_facts = self.memory_store.retrieve_all()
            retrieved_facts = self.retrieval.retrieve(stored_facts, question.question_text)
            
            retrieved_ids = {f.fact_id for f in retrieved_facts}
            required_ids = set(question.required_fact_ids)
            is_correct = required_ids.issubset(retrieved_ids)
            
            if is_correct:
                correct_count += 1
            
            question_results.append({
                'question_id': question.question_id,
                'correct': is_correct,
                'required_facts': list(required_ids),
                'retrieved_fact_ids': list(retrieved_ids)
            })
        
        return {
            'episode_id': episode.episode_id,
            'accuracy': correct_count / len(episode.questions),
            'correct_answers': correct_count,
            'total_questions': len(episode.questions),
            'memory_used': self.memory_store.get_size(),
            'memory_capacity': self.memory_store.capacity,
            'write_count': self.memory_store.write_count,
            'question_results': question_results,
            'memory_stats': self.memory_store.get_stats()
        }


class BaselineComparison:
    """
    Compares learned agent against multiple baselines.
    """
    
    def __init__(self, memory_capacity: int = 5, retrieval_budget: int = 3):
        """
        Initialize baseline comparison.
        
        Args:
            memory_capacity: Memory size for all agents
            retrieval_budget: Retrieval budget for all agents
        """
        self.memory_capacity = memory_capacity
        self.retrieval_budget = retrieval_budget
        
        self.baselines = {
            'no_memory': BaselineAgent(NoMemoryPolicy(), memory_capacity, retrieval_budget),
            'random': BaselineAgent(RandomPolicy(write_probability=0.5), memory_capacity, retrieval_budget),
            'heuristic': BaselineAgent(HeuristicPolicy(), memory_capacity, retrieval_budget)
        }
        
        self.results = {}
    
    def evaluate_baselines(self, episodes: List[Episode]) -> Dict[str, Dict]:
        """
        Evaluate all baselines on a set of episodes.
        
        Args:
            episodes: List of episodes to evaluate on
            
        Returns:
            Dictionary mapping baseline names to their results
        """
        results = {}
        
        for baseline_name, baseline_agent in self.baselines.items():
            baseline_results = {
                'name': baseline_name,
                'accuracies': [],
                'memory_efficiencies': []
            }
            
            for episode in episodes:
                metrics = baseline_agent.process_episode(episode)
                baseline_results['accuracies'].append(metrics['accuracy'])
                
                efficiency = metrics['memory_used'] / metrics['memory_capacity']
                baseline_results['memory_efficiencies'].append(efficiency)
            
            # Calculate summary statistics
            accuracies = baseline_results['accuracies']
            baseline_results['mean_accuracy'] = sum(accuracies) / len(accuracies)
            baseline_results['min_accuracy'] = min(accuracies)
            baseline_results['max_accuracy'] = max(accuracies)
            
            efficiencies = baseline_results['memory_efficiencies']
            baseline_results['mean_memory_efficiency'] = sum(efficiencies) / len(efficiencies)
            
            results[baseline_name] = baseline_results
        
        self.results = results
        return results
    
    def get_comparison_summary(self) -> Dict:
        """
        Get summary comparing all baselines.
        
        Returns:
            Dictionary with comparative metrics
        """
        if not self.results:
            return {}
        
        summary = {}
        for name, results in self.results.items():
            summary[name] = {
                'mean_accuracy': results.get('mean_accuracy', 0.0),
                'min_accuracy': results.get('min_accuracy', 0.0),
                'max_accuracy': results.get('max_accuracy', 0.0),
                'mean_memory_efficiency': results.get('mean_memory_efficiency', 0.0),
                'num_episodes': len(results.get('accuracies', []))
            }
        
        return summary
