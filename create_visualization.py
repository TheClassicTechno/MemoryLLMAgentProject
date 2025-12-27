#!/usr/bin/env python3
"""
Create budget vs accuracy visualization from baseline comparison results.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_budget_vs_accuracy():
    """
    Plot accuracy vs memory budget from baseline_comparison.json
    """
    results_dir = Path('results')
    
    # Try to load results (will retry if not ready)
    try:
        with open(results_dir / 'baseline_comparison.json') as f:
            comparison_data = json.load(f)
    except FileNotFoundError:
        print("Results not ready yet. Waiting...")
        import time
        time.sleep(10)
        with open(results_dir / 'baseline_comparison.json') as f:
            comparison_data = json.load(f)
    
    # Try to load budget scaling results
    try:
        with open(results_dir / 'budget_scaling.json') as f:
            budget_data = json.load(f)
    except FileNotFoundError:
        budget_data = None
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Comparison & Budget Scaling Analysis', fontsize=16, fontweight='bold')
    
    # 1. Baseline comparison (top-left)
    ax = axes[0, 0]
    baselines = list(comparison_data.keys())
    accuracies = [comparison_data[b]['mean'] for b in baselines]
    stds = [comparison_data[b]['std'] for b in baselines]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(baselines, accuracies, yerr=stds, capsize=5, color=colors[:len(baselines)], alpha=0.7)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Baseline Comparison (100 episodes each)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10)
    
    # 2. Budget scaling (top-right)
    ax = axes[0, 1]
    if budget_data and 'results' in budget_data:
        budgets = budget_data.get('budgets', [])
        results_by_baseline = budget_data['results']
        for baseline, color in zip(['Heuristic', 'Recency', 'Embedding', 'Random'], 
                                   colors[:4]):
            if baseline in results_by_baseline:
                accs = results_by_baseline[baseline]
                ax.plot(budgets, accs, marker='o', label=baseline, linewidth=2, 
                       markersize=8, color=color)
        
        ax.set_xlabel('Memory Budget (K)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Budget Scaling Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # 3. Ranking (bottom-left)
    ax = axes[1, 0]
    ranking = sorted(enumerate(baselines), key=lambda x: accuracies[x[0]], reverse=True)
    names = [baselines[i] for i, _ in ranking]
    scores = [accuracies[i] for i, _ in ranking]
    
    bars = ax.barh(names, scores, color=colors[:len(baselines)], alpha=0.7)
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_title('Baseline Ranking', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {score:.1%}', ha='left', va='center', fontsize=10)
    
    # 4. Summary table (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = "SUMMARY STATISTICS\n" + "="*40 + "\n\n"
    
    # Baseline comparison summary
    summary_text += "Baseline Comparison (100 episodes):\n"
    for baseline in baselines:
        data = comparison_data[baseline]
        summary_text += (f"{baseline:15} {data['mean']:.1%} "
                        f"± {data['std']:.1%}\n")
    
    if budget_data and 'results' in budget_data:
        summary_text += "\nBudget Scaling (Budget=5):\n"
        budgets = budget_data.get('budgets', [])
        results_by_baseline = budget_data['results']
        if budgets and len(budgets) > 0:
            for baseline in ['Heuristic', 'Recency', 'Embedding', 'Random']:
                if baseline in results_by_baseline and len(results_by_baseline[baseline]) > 0:
                    acc = results_by_baseline[baseline][0]  # First budget=5
                    summary_text += (f"{baseline:15} {acc:.1%}\n")
    
    summary_text += "\nKey Findings:\n"
    best_baseline = baselines[np.argmax(accuracies)]
    summary_text += f"• Best baseline: {best_baseline}\n"
    summary_text += f"• Accuracy range: {min(accuracies):.1%} - {max(accuracies):.1%}\n"
    heuristic_acc = comparison_data.get('Heuristic', {}).get('mean', 0)
    summary_text += f"• Heuristic baseline: {heuristic_acc:.1%}\n"
    if best_baseline != 'Heuristic':
        improvement = (comparison_data[best_baseline]['mean'] - heuristic_acc) / heuristic_acc * 100
        summary_text += f"• Improvement: +{improvement:.1f}%\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'budget_vs_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to results/budget_vs_accuracy.png")
    plt.close()


if __name__ == '__main__':
    plot_budget_vs_accuracy()
