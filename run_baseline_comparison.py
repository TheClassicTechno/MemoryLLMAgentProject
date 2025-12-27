#!/usr/bin/env python3
"""
Evaluate stronger baselines against heuristic and random.
Creates comparison plots and JSON results.
"""
import json
import numpy as np
from pathlib import Path
from env.streaming_qa_env import StreamingQAEnvironment
from agent.core_agent import MemoryAgent
from agent.policy import MemorySelectionPolicy, RandomPolicy
from baselines.stronger_baselines import (
    RecencyBaseline, FrequencyBaseline, EmbeddingSimilarityBaseline, HeuristicBaseline
)


def run_baseline_comparison(episodes: int = 100, memory_capacity: int = 5, retrieval_budget: int = 3):
    """
    Evaluate multiple baselines on the synthetic QA task.
    """
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON STUDY")
    print(f"{'='*60}\n")
    
    env = StreamingQAEnvironment(
        num_facts=10,
        num_questions=2,
        delay_length=1,
        distractor_density=0.3
    )
    
    # Initialize all baselines
    baselines = {
        'Heuristic': HeuristicBaseline(memory_capacity, retrieval_budget),
        'Recency': RecencyBaseline(memory_capacity, retrieval_budget),
        'Frequency': FrequencyBaseline(memory_capacity, retrieval_budget),
        'Embedding': EmbeddingSimilarityBaseline(memory_capacity, retrieval_budget),
        'Random': MemoryAgent(memory_capacity, retrieval_budget, 
                            policy=RandomPolicy()),
    }
    
    results = {}
    
    # Run evaluation for each baseline
    for baseline_name, baseline in baselines.items():
        print(f"Evaluating {baseline_name}... ", end="", flush=True)
        accuracies = []
        
        for ep in range(episodes):
            if ep % 20 == 0:
                print(f"{ep}.", end="", flush=True)
            
            episode = env.generate_episode(episode_id=ep)
            
            if hasattr(baseline, 'process_episode'):
                # Baselines with episode processing
                metrics = baseline.process_episode(episode)
                accuracy = metrics['accuracy']
            else:
                # Agent-based baselines
                baseline.memory_store.clear()
                correctness = []
                
                for fact in episode.facts:
                    baseline.process_fact(fact)
                
                for question in episode.questions:
                    answer, retrieved = baseline.answer_question(question)
                    required_ids = episode.get_required_fact_ids(question)
                    retrieved_ids = set(f.fact_id for f in retrieved)
                    correct = required_ids.issubset(retrieved_ids)
                    correctness.append(1.0 if correct else 0.0)
                
                accuracy = np.mean(correctness) if correctness else 0.0
            
            accuracies.append(accuracy)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        results[baseline_name] = {
            'mean': float(mean_acc),
            'std': float(std_acc),
            'accuracies': accuracies
        }
        
        print(f" {mean_acc:.1%} ± {std_acc:.1%}")
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    # Print comparison table
    print(f"{'Baseline':<15} {'Accuracy':<15} {'Std Dev':<15}")
    print("-" * 45)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for name, data in sorted_results:
        print(f"{name:<15} {data['mean']:>6.1%}       {data['std']:>6.1%}")
    
    # Calculate improvements
    best = sorted_results[0]
    heuristic = results.get('Heuristic', {}).get('mean', 0)
    
    print(f"\n{'='*60}")
    print(f"Best: {best[0]} at {best[1]['mean']:.1%}")
    print(f"Heuristic: {heuristic:.1%}")
    print(f"Improvement over heuristic: {(best[1]['mean'] - heuristic):.1%}")
    print(f"{'='*60}\n")
    
    return results


def create_budget_scaling_plot(memory_budgets=[5, 10, 20]):
    """
    Test baselines across different memory budgets.
    """
    print(f"\n{'='*60}")
    print(f"BUDGET SCALING STUDY")
    print(f"{'='*60}\n")
    
    env = StreamingQAEnvironment(num_facts=10, num_questions=2, 
                                delay_length=1, distractor_density=0.3)
    
    budget_results = {
        'Heuristic': [],
        'Recency': [],
        'Embedding': [],
        'Random': []
    }
    
    for budget in memory_budgets:
        print(f"Testing memory budget K={budget}... ", end="", flush=True)
        
        # Test heuristic
        accuracies = []
        for i in range(30):  # 30 episodes per budget
            episode = env.generate_episode(episode_id=i)
            baseline = HeuristicBaseline(memory_capacity=budget, retrieval_budget=3)
            metrics = baseline.process_episode(episode)
            accuracies.append(metrics['accuracy'])
        budget_results['Heuristic'].append(np.mean(accuracies))
        
        # Test recency
        accuracies = []
        for i in range(30):
            episode = env.generate_episode(episode_id=100+i)
            baseline = RecencyBaseline(memory_capacity=budget, retrieval_budget=3)
            metrics = baseline.process_episode(episode)
            accuracies.append(metrics['accuracy'])
        budget_results['Recency'].append(np.mean(accuracies))
        
        # Test embedding
        accuracies = []
        for i in range(30):
            episode = env.generate_episode(episode_id=200+i)
            baseline = EmbeddingSimilarityBaseline(memory_capacity=budget, retrieval_budget=3)
            metrics = baseline.process_episode(episode)
            accuracies.append(metrics['accuracy'])
        budget_results['Embedding'].append(np.mean(accuracies))
        
        # Test random
        accuracies = []
        for i in range(30):
            episode = env.generate_episode(episode_id=300+i)
            agent = MemoryAgent(memory_capacity=budget, retrieval_budget=3,
                              policy=RandomPolicy())
            metrics = agent.process_episode(episode)
            accuracies.append(metrics['accuracy'])
        budget_results['Random'].append(np.mean(accuracies))
        
        print("Done")
    
    # Save results
    results_dir = Path('results')
    with open(results_dir / 'budget_scaling.json', 'w') as f:
        json.dump({
            'budgets': memory_budgets,
            'results': budget_results
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BUDGET SCALING RESULTS")
    print(f"{'='*60}\n")
    print(f"{'Budget':<10} {'Heuristic':<12} {'Recency':<12} {'Embedding':<12} {'Random':<12}")
    print("-" * 58)
    
    for i, budget in enumerate(memory_budgets):
        print(f"{budget:<10} {budget_results['Heuristic'][i]:.1%}       "
              f"{budget_results['Recency'][i]:.1%}       "
              f"{budget_results['Embedding'][i]:.1%}       "
              f"{budget_results['Random'][i]:.1%}")
    
    print(f"\n{'='*60}\n")
    
    return budget_results


if __name__ == '__main__':
    # Run baseline comparison (100 episodes)
    comparison_results = run_baseline_comparison(episodes=100)
    
    # Run budget scaling study
    budget_results = create_budget_scaling_plot(memory_budgets=[5, 10, 20])
    
    print("Results saved to results/baseline_comparison.json and results/budget_scaling.json")
