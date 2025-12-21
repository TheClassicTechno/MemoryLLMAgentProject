"""
Core agent implementation for streaming QA task.
"""
from typing import List, Tuple, Optional
from env.streaming_qa_env import Fact, Question, Episode
from memory.memory_store import MemoryStore
from memory.retrieval import KeywordRetrieval
from agent.policy import MemorySelectionPolicy
from agent.llm_adapter import LLMAdapter, LocalMockAdapter


class MemoryAgent:
    """
    Agent that learns what to remember in a streaming QA task.
    
    Uses LLM for answer generation and learned policy for memory selection.
    """
    
    def __init__(self, memory_capacity: int = 5, retrieval_budget: int = 3,
                 policy: Optional[MemorySelectionPolicy] = None,
                 llm: Optional[LLMAdapter] = None):
        """
        Initialize the agent.
        
        Args:
            memory_capacity: Maximum number of facts to store
            retrieval_budget: Maximum facts to retrieve for a query
            policy: Memory selection policy (defaults to learned policy)
            llm: LLM adapter for answer generation (defaults to mock)
        """
        self.memory_store = MemoryStore(capacity=memory_capacity)
        self.retrieval = KeywordRetrieval(top_k=retrieval_budget)
        self.policy = policy or MemorySelectionPolicy()
        self.llm = llm or LocalMockAdapter()
        
        # Tracking
        self.decision_history: List[Tuple[Fact, bool]] = []
        self.reward_history: List[float] = []
    
    def process_fact(self, fact: Fact) -> bool:
        """
        Process an incoming fact and decide whether to store it.
        
        Args:
            fact: The fact to process
            
        Returns:
            True if fact was stored, False otherwise
        """
        decision = self.policy.decide(fact)
        
        if decision:
            self.memory_store.write(fact)
        
        self.decision_history.append((fact, decision))
        
        return decision
    
    def answer_question(self, question: Question) -> Tuple[str, List[Fact]]:
        """
        Answer a question using LLM with retrieved memories.
        
        Args:
            question: The question to answer
            
        Returns:
            Tuple of (answer_text, retrieved_facts)
        """
        import time
        
        # Retrieve relevant facts from memory
        stored_facts = self.memory_store.retrieve_all()
        retrieved_facts = self.retrieval.retrieve(stored_facts, question.question_text)
        
        # Format context from retrieved facts
        if retrieved_facts:
            context = "\n".join([f.content for f in retrieved_facts])
        else:
            context = None
        
        # Longer delay to avoid API rate limiting (2 seconds between requests)
        time.sleep(2.0)
        
        # Use LLM to generate answer
        answer = self.llm.answer_question(question.question_text, context=context)
        
        return answer, retrieved_facts
    
    def process_episode(self, episode: Episode) -> dict:
        """
        Process a complete episode and return metrics.
        
        Args:
            episode: The episode to process
            
        Returns:
            Dictionary with episode metrics
        """
        self.memory_store.clear()
        self.decision_history.clear()
        self.reward_history.clear()
        
        # Phase 1: Observe facts
        for fact in episode.facts:
            self.process_fact(fact)
        
        # Phase 2: Delay period (no action needed)
        
        # Phase 3: Answer questions
        correct_count = 0
        total_count = len(episode.questions)
        
        question_results = []
        
        for question in episode.questions:
            answer, retrieved_facts = self.answer_question(question)
            
            # Evaluate correctness
            is_correct = self._evaluate_answer(question, retrieved_facts, answer)
            correct_count += 1 if is_correct else 0
            
            question_results.append({
                'question_id': question.question_id,
                'question_text': question.question_text,
                'correct': is_correct,
                'answer': answer,
                'required_facts': question.required_fact_ids,
                'retrieved_fact_ids': [f.fact_id for f in retrieved_facts]
            })
        
        # Assign rewards based on correctness
        reward_signal = correct_count / total_count
        self._assign_rewards(reward_signal)
        
        # Update policy if learnable
        if hasattr(self.policy, 'update_episode'):
            self.policy.update_episode(self.decision_history, self.reward_history)
        
        metrics = {
            'episode_id': episode.episode_id,
            'accuracy': correct_count / total_count,
            'correct_answers': correct_count,
            'total_questions': total_count,
            'memory_used': self.memory_store.get_size(),
            'memory_capacity': self.memory_store.capacity,
            'write_count': self.memory_store.write_count,
            'question_results': question_results,
            'memory_stats': self.memory_store.get_stats()
        }
        
        return metrics
    
    def _evaluate_answer(self, question: Question, retrieved_facts: List[Fact], 
                        answer: str) -> bool:
        """
        Evaluate if an answer is correct.
        
        Uses similarity scoring for LLM-generated answers.
        
        Args:
            question: The question that was answered
            retrieved_facts: Facts the agent retrieved
            answer: The agent's answer
            
        Returns:
            True if answer is correct
        """
        from eval.metrics import similarity_score
        
        # Check if all required facts were retrieved
        retrieved_ids = {f.fact_id for f in retrieved_facts}
        required_ids = set(question.required_fact_ids)
        
        has_all_required = required_ids.issubset(retrieved_ids)
        
        if not has_all_required:
            return False
        
        # Score answer similarity to expected answer
        sim = similarity_score(answer, question.answer_text)
        return sim >= 0.8
    
    def _assign_rewards(self, global_reward: float):
        """
        Assign rewards to each decision made during the episode.
        
        Args:
            global_reward: Overall episode reward signal
        """
        self.reward_history = [global_reward] * len(self.decision_history)
    
    def reset_episode(self):
        """Reset agent state for new episode."""
        self.memory_store.clear()
        self.decision_history.clear()
        self.reward_history.clear()
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        return self.memory_store.get_stats()
