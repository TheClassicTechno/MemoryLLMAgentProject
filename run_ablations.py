#!/usr/bin/env python3
"""
Quick ablation studies for Phase 1 results.
Skips API delays for fast iteration.
"""
import json
from env.streaming_qa_env import StreamingQAEnvironment
from agent.core_agent import MemoryAgent
from agent.policy import RandomPolicy, HeuristicPolicy
from agent.llm_adapter import get_llm_adapter

def run_ablation(name: str, config: dict, num_episodes: int = 30):
    """Run a single ablation study."""
    results = {}
    env = StreamingQAEnvironment(
        num_facts=config.get('facts', 10),
        num_questions=config.get('questions', 2),
        delay_length=config.get('delay', 3),
        distractor_density=config.get('distractor_density', 0.3)
    )
    
    for policy_name, policy_class in [('heuristic', HeuristicPolicy), ('random', RandomPolicy)]:
        accuracies = []
        
        for episode_id in range(num_episodes):
            episode = env.generate_episode(episode_id)
            agent = MemoryAgent(
                memory_capacity=config.get('memory', 5),
                retrieval_budget=3,
                policy=policy_class(),
                llm=get_llm_adapter('mock')
            )
            metrics = agent.process_episode(episode)
            accuracies.append(metrics['accuracy'])
        
        mean_acc = sum(accuracies) / len(accuracies)
        results[policy_name] = mean_acc
        print(f"  {policy_name}: {mean_acc:.3f}")
    
    return results

# Study 1: Memory Size Sweep
print("\n=== Ablation 1: Memory Size Sweep ===")
memory_results = {}
for k in [2, 4, 8, 16]:
    print(f"Memory capacity: {k}")
    memory_results[f"k_{k}"] = run_ablation(f"memory_k{k}", {'memory': k})

with open('results/ablation_memory_size.json', 'w') as f:
    json.dump(memory_results, f, indent=2)
print("Saved to results/ablation_memory_size.json")

# Study 2: Delay Sweep
print("\n=== Ablation 2: Delay Sweep ===")
delay_results = {}
for delay in [1, 3, 5]:
    print(f"Delay length: {delay}")
    delay_results[f"delay_{delay}"] = run_ablation(f"delay_{delay}", {'delay': delay})

with open('results/ablation_delay.json', 'w') as f:
    json.dump(delay_results, f, indent=2)
print("Saved to results/ablation_delay.json")

# Study 3: Distractor Density Sweep
print("\n=== Ablation 3: Distractor Density Sweep ===")
distractor_results = {}
for density in [0.1, 0.3, 0.5]:
    print(f"Distractor density: {density}")
    distractor_results[f"density_{density}"] = run_ablation(f"density_{density}", {'distractor_density': density})

with open('results/ablation_distractor_density.json', 'w') as f:
    json.dump(distractor_results, f, indent=2)
print("Saved to results/ablation_distractor_density.json")

print("\n=== All ablations complete ===")
