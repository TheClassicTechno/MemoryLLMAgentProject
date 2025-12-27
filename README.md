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
