#!/usr/bin/env python3
"""
Debug script to diagnose why accuracy is 0.00
"""
import sys
sys.path.insert(0, '/Users/julih/Documents/MemoryLLMAgentProject')

from env.streaming_qa_env import StreamingQAEnvironment
from agent.core_agent import MemoryAgent
from agent.llm_adapter import get_llm_adapter

# Create environment
env = StreamingQAEnvironment(
    num_facts=3,
    num_questions=1,
    delay_length=3,
    distractor_density=0.3
)

# Create agent with Ollama
print("Initializing agent with Ollama...")
agent = MemoryAgent(
    memory_capacity=5,
    retrieval_budget=3,
    llm=get_llm_adapter('ollama')
)

# Generate one episode
print("\nGenerating episode...")
episode = env.generate_episode(episode_id=0)

print(f"\n=== EPISODE DATA ===")
print(f"Facts: {len(episode.facts)}")
for i, fact in enumerate(episode.facts):
    print(f"  Fact {i}: ID={fact.fact_id}, content='{fact.content}'")

print(f"\nQuestions: {len(episode.questions)}")
for i, q in enumerate(episode.questions):
    print(f"  Question {i}: '{q.question_text}'")
    print(f"    Expected answer: '{q.answer_text}'")
    print(f"    Required fact IDs: {q.required_fact_ids}")

# Process episode with debug output
print(f"\n=== PROCESSING EPISODE ===")

# Process facts
print("\nPhase 1: Processing facts...")
for fact in episode.facts:
    decision = agent.policy.decide(fact)
    print(f"  Fact '{fact.content}' (ID={fact.fact_id})")
    print(f"    Feature vector: {agent.policy.feature_extractor.extract_features(fact)}")
    print(f"    Decision: {'STORE' if decision else 'SKIP'}")
    if decision:
        agent.memory_store.write(fact)

print(f"\n  Memory after facts: {agent.memory_store.get_stats()}")

# Answer questions
print("\nPhase 2: Answering questions...")
for question in episode.questions:
    print(f"\n  Question: '{question.question_text}'")
    print(f"  Required fact IDs: {question.required_fact_ids}")
    
    # Check what's in memory
    all_facts = agent.memory_store.read_all()
    print(f"  Facts in memory: {len(all_facts)}")
    for f in all_facts:
        print(f"    - {f.fact_id}: '{f.content}'")
    
    # Retrieve facts
    retrieved = agent.memory_store.retrieve(question.question_text, budget=3)
    print(f"  Retrieved facts: {len(retrieved)}")
    for f in retrieved:
        print(f"    - {f.fact_id}: '{f.content}'")
    
    # Check if required facts were retrieved
    retrieved_ids = {f.fact_id for f in retrieved}
    required_ids = set(question.required_fact_ids)
    has_all = required_ids.issubset(retrieved_ids)
    print(f"  Has all required facts? {has_all}")
    
    # Get answer
    if retrieved:
        context = "\n".join([f.content for f in retrieved])
    else:
        context = None
    
    answer = agent.llm.answer_question(question.question_text, context=context)
    print(f"  LLM answer: '{answer}'")
    print(f"  Expected answer: '{question.answer_text}'")
    
    # Evaluate
    is_correct = agent._evaluate_answer(question, retrieved, answer)
    print(f"  Correct? {is_correct}")

print("\n=== DIAGNOSIS ===")
print("If accuracy is 0.00, check:")
print("1. Are facts being stored in memory?")
print("2. Are required facts being retrieved?")
print("3. Is the LLM providing reasonable answers?")
