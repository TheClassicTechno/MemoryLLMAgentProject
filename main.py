"""
Main experiment runner for the memory learning project.
"""
import json
import argparse
from typing import Dict, List
from env.streaming_qa_env import StreamingQAEnvironment
from agent.core_agent import MemoryAgent
from agent.policy import MemorySelectionPolicy, RandomPolicy, HeuristicPolicy
from agent.llm_adapter import get_llm_adapter
from eval.metrics import MetricsAggregator, evaluate_episode
from eval.baselines import BaselineComparison
from utils.config import save_json, get_results_path
from utils.analysis import ResultsAnalyzer


class ExperimentRunner:
    """
    Runs the full memory learning experiment with LLM integration.
    """
    
    def __init__(self, num_episodes: int = 10, num_facts: int = 10,
                 num_questions: int = 2, memory_capacity: int = 5,
                 retrieval_budget: int = 3, llm_type: str = "mock",
                 llm_kwargs: Dict = None):
        """
        Initialize experiment runner.
        
        Args:
            num_episodes: Number of episodes to run
            num_facts: Number of facts per episode
            num_questions: Number of questions per episode
            memory_capacity: Maximum facts in memory
            retrieval_budget: Max facts to retrieve per query
            llm_type: Type of LLM adapter ("mock", "openai", "anthropic")
            llm_kwargs: Additional kwargs for LLM adapter
        """
        self.num_episodes = num_episodes
        self.num_facts = num_facts
        self.num_questions = num_questions
        self.memory_capacity = memory_capacity
        self.retrieval_budget = retrieval_budget
        self.llm_type = llm_type
        
        self.env = StreamingQAEnvironment(
            num_facts=num_facts,
            num_questions=num_questions,
            delay_length=3,
            distractor_density=0.3
        )
        
        # Create LLM adapter
        try:
            self.llm = get_llm_adapter(llm_type, **(llm_kwargs or {}))
            llm_info = f"{llm_type} LLM"
        except Exception as e:
            print(f"Warning: Could not initialize {llm_type} adapter: {e}")
            print("Falling back to mock adapter")
            self.llm = get_llm_adapter("mock")
            llm_info = "mock LLM (local)"
        
        self.learned_agent = MemoryAgent(
            memory_capacity=memory_capacity,
            retrieval_budget=retrieval_budget,
            policy=MemorySelectionPolicy(),
            llm=self.llm
        )
        
        self.metrics_aggregator = MetricsAggregator()
        self.results = {}
        self.llm_info = llm_info
    
    def run_learned_agent(self) -> Dict:
        """
        Run the learned agent on all episodes.
        
        Returns:
            Dictionary with learned agent results
        """
        print(f"Running learned agent for {self.num_episodes} episodes...")
        
        accuracies = []
        memory_efficiencies = []
        write_counts = []
        
        for episode_id in range(self.num_episodes):
            episode = self.env.generate_episode(episode_id)
            metrics = self.learned_agent.process_episode(episode)
            
            accuracies.append(metrics['accuracy'])
            efficiency = metrics['memory_used'] / metrics['memory_capacity']
            memory_efficiencies.append(efficiency)
            write_counts.append(metrics['write_count'])
            
            # Evaluate episode
            eval_result = evaluate_episode(metrics)
            
            print(f"  Episode {episode_id}: Accuracy {metrics['accuracy']:.2f}, Memory efficiency {efficiency:.2f}")
        
        results = {
            'accuracies': accuracies,
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'memory_efficiencies': memory_efficiencies,
            'mean_memory_efficiency': sum(memory_efficiencies) / len(memory_efficiencies),
            'write_counts': write_counts,
            'mean_writes_per_episode': sum(write_counts) / len(write_counts),
            'num_episodes': self.num_episodes
        }
        
        self.results['learned_agent'] = results
        return results
    
    def run_baselines(self) -> Dict:
        """
        Run baseline agents for comparison.
        
        Returns:
            Dictionary with baseline results
        """
        print(f"Running baseline agents for {self.num_episodes} episodes...")
        
        # Generate episodes
        episodes = [self.env.generate_episode(i) for i in range(self.num_episodes)]
        
        # Run baselines
        baseline_comparison = BaselineComparison(
            memory_capacity=self.memory_capacity,
            retrieval_budget=self.retrieval_budget
        )
        
        baseline_results = baseline_comparison.evaluate_baselines(episodes)
        
        # Print results
        summary = baseline_comparison.get_comparison_summary()
        for name, stats in summary.items():
            print(f"  {name}: Mean accuracy {stats['mean_accuracy']:.2f}, Memory efficiency {stats['mean_memory_efficiency']:.2f}")
        
        self.results['baseline_comparison'] = baseline_results
        return baseline_results
    
    def run_experiment(self) -> Dict:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary with all results
        """
        print("Starting experiment...")
        print(f"Configuration: {self.num_episodes} episodes, {self.num_facts} facts, {self.num_questions} questions")
        print(f"Memory: capacity={self.memory_capacity}, retrieval_budget={self.retrieval_budget}")
        print(f"LLM: {self.llm_info}")
        print()
        
        # Run learned agent
        self.run_learned_agent()
        print()
        
        # Run baselines
        self.run_baselines()
        print()
        
        print("Experiment complete!")
        return self.results
    
    def save_results(self, output_path: str = None):
        """
        Save results to JSON file.
        
        Args:
            output_path: Path to save results (default: results/experiment_results.json)
        """
        if not output_path:
            output_path = get_results_path('experiment_results.json')
        
        save_json(self.results, output_path)
        print(f"Results saved to {output_path}")
    
    def generate_plots(self, output_dir: str = None):
        """
        Generate visualization plots.
        
        Args:
            output_dir: Directory to save plots (default: results/)
        """
        if not output_dir:
            output_dir = get_results_path()
        
        analyzer = ResultsAnalyzer(self.results)
        
        # Generate plots
        learning_curve_path = analyzer.plot_learning_curve(f'{output_dir}/learning_curve.png')
        comparison_path = analyzer.plot_baseline_comparison(f'{output_dir}/baseline_comparison.png')
        efficiency_path = analyzer.plot_memory_efficiency(f'{output_dir}/memory_efficiency.png')
        
        print(f"Generated learning curve plot: {learning_curve_path}")
        print(f"Generated baseline comparison plot: {comparison_path}")
        print(f"Generated memory efficiency plot: {efficiency_path}")
        
        # Generate report
        report = analyzer.generate_summary_report()
        report_path = get_results_path('experiment_report.json')
        save_json(report, report_path)
        print(f"Generated summary report: {report_path}")


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description='Run memory learning experiment with LLM agent')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--facts', type=int, default=10, help='Facts per episode')
    parser.add_argument('--questions', type=int, default=2, help='Questions per episode')
    parser.add_argument('--memory', type=int, default=5, help='Memory capacity')
    parser.add_argument('--retrieval', type=int, default=3, help='Retrieval budget')
    parser.add_argument('--llm', type=str, default='mock', 
                       choices=['mock', 'openai', 'anthropic', 'ollama'],
                       help='LLM adapter type')
    parser.add_argument('--output', type=str, default=None, help='Output results file')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = ExperimentRunner(
        num_episodes=args.episodes,
        num_facts=args.facts,
        num_questions=args.questions,
        memory_capacity=args.memory,
        retrieval_budget=args.retrieval,
        llm_type=args.llm
    )
    
    results = runner.run_experiment()
    runner.save_results(args.output)
    
    if args.plots:
        runner.generate_plots()


if __name__ == '__main__':
    main()
