"""
Visualization and analysis utilities.
"""
import json
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ResultsAnalyzer:
    """
    Analyzes and visualizes experiment results.
    """
    
    def __init__(self, results: Dict):
        """
        Initialize analyzer.
        
        Args:
            results: Dictionary of results
        """
        self.results = results
    
    def plot_learning_curve(self, output_path: str = None) -> Optional[str]:
        """
        Plot learning curve over episodes.
        
        Args:
            output_path: Path to save plot
            
        Returns:
            Path to saved plot or None
        """
        if 'learned_agent' not in self.results:
            return None
        
        learned_results = self.results['learned_agent']
        accuracies = learned_results.get('accuracies', [])
        
        if not accuracies:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Episode Number')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve: Agent Accuracy Over Episodes')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.0])
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            return output_path
        
        return None
    
    def plot_baseline_comparison(self, output_path: str = None) -> Optional[str]:
        """
        Plot comparison of learned agent vs baselines.
        
        Args:
            output_path: Path to save plot
            
        Returns:
            Path to saved plot or None
        """
        comparison_data = self.results.get('baseline_comparison', {})
        
        if not comparison_data:
            return None
        
        names = []
        mean_accuracies = []
        
        for name, data in comparison_data.items():
            names.append(name)
            mean_accuracies.append(data.get('mean_accuracy', 0.0))
        
        if not names:
            return None
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, mean_accuracies, color=['steelblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylabel('Mean Accuracy')
        plt.title('Baseline Comparison: Mean Accuracy')
        plt.ylim([0, 1.0])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            return output_path
        
        return None
    
    def plot_memory_efficiency(self, output_path: str = None) -> Optional[str]:
        """
        Plot memory efficiency across agents.
        
        Args:
            output_path: Path to save plot
            
        Returns:
            Path to saved plot or None
        """
        comparison_data = self.results.get('baseline_comparison', {})
        
        if not comparison_data:
            return None
        
        names = []
        memory_efficiencies = []
        
        for name, data in comparison_data.items():
            names.append(name)
            memory_efficiencies.append(data.get('mean_memory_efficiency', 0.0))
        
        if not names:
            return None
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, memory_efficiencies, color=['steelblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylabel('Mean Memory Efficiency')
        plt.title('Memory Efficiency Comparison')
        plt.ylim([0, 1.0])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            return output_path
        
        return None
    
    def generate_summary_report(self) -> Dict:
        """
        Generate summary report of results.
        
        Returns:
            Dictionary with summary statistics
        """
        report = {}
        
        if 'learned_agent' in self.results:
            learned = self.results['learned_agent']
            report['learned_agent_summary'] = {
                'mean_accuracy': sum(learned.get('accuracies', [])) / max(len(learned.get('accuracies', [])), 1),
                'min_accuracy': min(learned.get('accuracies', []), default=0.0),
                'max_accuracy': max(learned.get('accuracies', []), default=0.0),
                'num_episodes': len(learned.get('accuracies', []))
            }
        
        if 'baseline_comparison' in self.results:
            report['baseline_comparison'] = self.results['baseline_comparison']
        
        return report
