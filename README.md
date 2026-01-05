# MemoryLLMAgentProject

Personal project by Juli Huang. Self-exploring concepts such as agent policy, memory, and environment.

In this project, I research how LLM-based agents should decide what to remember under strict memory and cost constraints, using online learning and controlled evaluation rather than heuristic memory systems.


## Datasets I used
I created a Synthetic Streaming QA dataset and used a small subset of the SQuAD dataset.
## Project Structure

### Phase 1: Baseline & Ablation Studies
- Established heuristic baseline (66.5% accuracy on 100 episodes)
- Conducted ablation studies across memory capacity, delay length, and distractor density
- Results: `/results/ablation_*.json`

### Phase 2 (Option B): Stronger Baselines & Real-Data Validation
- **Implemented 4 memory selection baselines:**
  - RecencyBaseline: Keep most recent K facts (FIFO strategy)
  - FrequencyBaseline: Keep most frequently referenced facts
  - EmbeddingSimilarityBaseline: Keep facts most similar to current query (Jaccard similarity)
  - HeuristicBaseline: Type-based filtering (Phase 1 reference)

- **Key Results:**
  - Embedding-based selection: **82.5% accuracy** (16% improvement over heuristic)
  - Budget scaling study: K={5,10,20} shows robust performance
  - SQuAD validation: 10/10 examples correct on real data

- **Deliverables:**
  - `baselines/stronger_baselines.py`: 4 baseline implementations
  - `run_baseline_comparison.py`: Evaluation harness with statistical reporting
  - `create_visualization.py`: Publication-quality comparison plots
  - `squad_validation.py`: Real-data sanity check on SQuAD subset
  - `results/budget_vs_accuracy.png`: Visualization of baseline performance

## Key Files

- `agent/core_agent.py`: Main agent implementation with memory management
- `agent/policy.py`: Memory selection policy implementations (heuristic, random, learned)
- `env/streaming_qa_env.py`: Streaming QA task environment generator
- `memory/memory_store.py`: Memory storage backend with capacity constraints
- `memory/retrieval.py`: Keyword retrieval module

## Running Experiments

```bash
# Run baseline comparison (100 episodes per baseline)
python3 run_baseline_comparison.py

# Generate visualization
python3 create_visualization.py

# Validate on SQuAD
python3 squad_validation.py

# Check ablation results
python3 check_ablation_results.py
```

## Results Summary

| Baseline | Accuracy | Notes |
|----------|----------|-------|
| Embedding (best) | 82.5% | Query-aware semantic matching |
| Heuristic | 66.5% | Type-based filtering |
| Frequency | 53.0% | Usage-based selection |
| Recency | 48.5% | FIFO strategy |
| Random | 43.0% | Negative control |

**Budget Scaling (K=5,10,20):** Embedding maintains 76-82% across all budgets, showing robustness to memory constraints.



## Common Questions and Answers

**Q1: Why not just use embedding similarity?**

**Answer:** Embedding similarity is one of my strongest baselines, achieving 82.5% accuracy. However, i implemented multiple baseline comparisons to understand when different memory selection strategies do well. results show that embedding similarity outperforms recency (48.5%), frequency (53.0%), and type-based heuristics (66.5%), but each strategy has different computational costs and use cases. The goal is to establish which selection method works best under memory constraints.

**Q2: Why compare multiple baselines instead of just picking one strategy?**

**Answer:** Different memory selection strategies make different trade-offs. Recency-based selection (FIFO) is computationally cheapest but performs poorly (48.5%). Embedding similarity is most accurate (82.5%) but requires computing similarity scores for every query. Frequency-based selection captures fact importance but doesn't adapt to query context. By comparing these baselines across controlled conditions, we can make more accurate and informed design choices for memory-constrained LLM agents.


**Q3: What does the evaluation techniques actually measure?**

**Answer:** I evaluate how different memory selection strategies perform under strict memory budgets (K=5, 10, 20 facts) on a streaming QA task. The agent must decide which facts to keep or discard as new information arrives, then answer questions using only its limited memory. i measure accuracy across 100 episodes on synthetic data and validate on a SQuAD subset to ensure the results generalize to real question-answering scenarios.


**Q4: Why test on both synthetic and real data?**

**Answer:** Synthetic data gives controlled evaluation with known distributions. we can systematically vary memory capacity (limit on what info it can hold), delay length (how long ago important info happened), and distractor density (how much junk/noise) to understand each baseline's behavior. SQuAD validation (10/10 examples correct with embedding baseline) confirms that my findings aren't artifacts of synthetic data generation and that the memory selection strategies work on real-world question-answering tasks.


**Q5: What did you learn from the budget scaling experiments?**

**Answer:** I tested memory budgets of K={5, 10, 20} facts across all baselines. Embedding similarity maintains 76-82% accuracy across all budgets, showing robustness to memory constraints. In contrast, recency and frequency baselines degrade significantly at lower budgets. This demonstrates that query-aware semantic matching is not just more accurate, but also more stable when memory is severely limited, a critical finding for resource-constrained deployments.


**Q6: What are the next steps?**

**Answer:** I currently establishes strong empirical baselines for memory selection. Future work could implement adaptive policies using reinforcement learning (e.g., contextual bandits) that learn to combine multiple signals: recency, frequency, and similarity-based on question type and context. 



