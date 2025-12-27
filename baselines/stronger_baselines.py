"""
Stronger baselines for memory selection in streaming QA.

Implements recency, frequency, and embedding-based strategies.
"""
import numpy as np
from typing import List, Tuple
from env.streaming_qa_env import Fact, Question, Episode
from memory.memory_store import MemoryStore
from memory.retrieval import KeywordRetrieval


class RecencyBaseline:
    """Always keep the most recent facts."""
    
    def __init__(self, memory_capacity: int = 5, retrieval_budget: int = 3):
        self.memory_store = MemoryStore(capacity=memory_capacity)
        self.retrieval = KeywordRetrieval(top_k=retrieval_budget)
        self.all_facts: List[Fact] = []
    
    def process_fact(self, fact: Fact) -> bool:
        """Store the fact (always keep recent)."""
        self.all_facts.append(fact)
        # Keep only most recent K facts
        if len(self.all_facts) > self.memory_store.capacity:
            self.memory_store.clear()
            for f in self.all_facts[-self.memory_store.capacity:]:
                self.memory_store.write(f)
        else:
            self.memory_store.write(fact)
        return True
    
    def answer_question(self, question: Question) -> Tuple[str, List[Fact]]:
        """Retrieve and answer question."""
        retrieved = self.retrieval.retrieve(self.memory_store.facts, question.question_text)
        answer = " ".join([f.content for f in retrieved])
        return answer, retrieved
    
    def process_episode(self, episode: Episode) -> dict:
        """Process episode and track accuracy."""
        self.memory_store.clear()
        self.all_facts = []
        
        correctness = []
        
        # Process facts
        for fact in episode.facts:
            self.process_fact(fact)
        
        # Answer questions
        for question in episode.questions:
            answer, retrieved = self.answer_question(question)
            # Check if all required facts were retrieved
            required_fact_ids = set(question.required_fact_ids) if question.required_fact_ids else set()
            retrieved_ids = set(f.fact_id for f in retrieved)
            correct = required_fact_ids.issubset(retrieved_ids)
            correctness.append(1.0 if correct else 0.0)
        
        accuracy = np.mean(correctness) if correctness else 0.0
        
        return {
            'accuracy': accuracy,
            'correctness': correctness,
            'memory_used': len(self.memory_store.facts) / self.memory_store.capacity
        }


class FrequencyBaseline:
    """Keep facts that are referenced most frequently in queries."""
    
    def __init__(self, memory_capacity: int = 5, retrieval_budget: int = 3):
        self.memory_store = MemoryStore(capacity=memory_capacity)
        self.retrieval = KeywordRetrieval(top_k=retrieval_budget)
        self.all_facts: List[Fact] = []
        self.fact_scores: dict = {}  # fact_id -> reference count
    
    def process_fact(self, fact: Fact) -> bool:
        """Add fact to history."""
        self.all_facts.append(fact)
        if fact.fact_id not in self.fact_scores:
            self.fact_scores[fact.fact_id] = 0
        return True
    
    def update_memory(self):
        """Update memory with highest-scoring facts."""
        self.memory_store.clear()
        # Sort facts by score, then by recency for ties
        sorted_facts = sorted(
            self.all_facts,
            key=lambda f: (self.fact_scores.get(f.fact_id, 0), self.all_facts.index(f)),
            reverse=True
        )
        for fact in sorted_facts[:self.memory_store.capacity]:
            self.memory_store.write(fact)
    
    def answer_question(self, question: Question) -> Tuple[str, List[Fact]]:
        """Retrieve and answer question, updating scores."""
        retrieved = self.retrieval.retrieve(self.memory_store.facts, question.question_text)
        
        # Update scores for retrieved facts
        for fact in retrieved:
            self.fact_scores[fact.fact_id] = self.fact_scores.get(fact.fact_id, 0) + 1
        
        answer = " ".join([f.content for f in retrieved])
        return answer, retrieved
    
    def process_episode(self, episode: Episode) -> dict:
        """Process episode and track accuracy."""
        self.memory_store.clear()
        self.all_facts = []
        self.fact_scores = {}
        
        correctness = []
        
        # Process facts
        for fact in episode.facts:
            self.process_fact(fact)
        
        # Answer questions, updating scores as we go
        for question in episode.questions:
            self.update_memory()  # Re-optimize memory based on current scores
            answer, retrieved = self.answer_question(question)
            
            required_fact_ids = set(question.required_fact_ids) if question.required_fact_ids else set()
            retrieved_ids = set(f.fact_id for f in retrieved)
            correct = required_fact_ids.issubset(retrieved_ids)
            correctness.append(1.0 if correct else 0.0)
        
        accuracy = np.mean(correctness) if correctness else 0.0
        
        return {
            'accuracy': accuracy,
            'correctness': correctness,
            'memory_used': len(self.memory_store.facts) / self.memory_store.capacity
        }


class EmbeddingSimilarityBaseline:
    """Keep facts most similar to the current/recent queries."""
    
    def __init__(self, memory_capacity: int = 5, retrieval_budget: int = 3):
        self.memory_store = MemoryStore(capacity=memory_capacity)
        self.retrieval = KeywordRetrieval(top_k=retrieval_budget)
        self.all_facts: List[Fact] = []
        self.recent_queries: List[str] = []
    
    def _simple_similarity(self, fact_text: str, query_text: str) -> float:
        """Simple bag-of-words cosine similarity."""
        # Tokenize
        fact_words = set(fact_text.lower().split())
        query_words = set(query_text.lower().split())
        
        if not fact_words or not query_words:
            return 0.0
        
        # Jaccard similarity as proxy for cosine
        intersection = len(fact_words & query_words)
        union = len(fact_words | query_words)
        return intersection / union if union > 0 else 0.0
    
    def process_fact(self, fact: Fact) -> bool:
        """Add fact to history."""
        self.all_facts.append(fact)
        return True
    
    def update_memory(self, current_query: str):
        """Update memory with facts most similar to current query."""
        self.memory_store.clear()
        
        if not self.all_facts:
            return
        
        # Score each fact by similarity to current query
        scores = []
        for fact in self.all_facts:
            # Use content for similarity
            similarity = self._simple_similarity(fact.content, current_query)
            # Also boost recent facts slightly
            recency_boost = 0.1 * (1.0 - (len(self.all_facts) - self.all_facts.index(fact)) / len(self.all_facts))
            total_score = similarity + recency_boost
            scores.append((total_score, fact))
        
        # Keep top-K facts
        sorted_facts = sorted(scores, key=lambda x: x[0], reverse=True)
        for score, fact in sorted_facts[:self.memory_store.capacity]:
            self.memory_store.write(fact)
    
    def answer_question(self, question: Question) -> Tuple[str, List[Fact]]:
        """Retrieve and answer question."""
        self.recent_queries.append(question.question_text)
        self.update_memory(question.question_text)
        
        retrieved = self.retrieval.retrieve(self.memory_store.facts, question.question_text)
        answer = " ".join([f.content for f in retrieved])
        return answer, retrieved
    
    def process_episode(self, episode: Episode) -> dict:
        """Process episode and track accuracy."""
        self.memory_store.clear()
        self.all_facts = []
        self.recent_queries = []
        
        correctness = []
        
        # Process facts
        for fact in episode.facts:
            self.process_fact(fact)
        
        # Answer questions
        for question in episode.questions:
            answer, retrieved = self.answer_question(question)
            
            required_fact_ids = set(question.required_fact_ids) if question.required_fact_ids else set()
            retrieved_ids = set(f.fact_id for f in retrieved)
            correct = required_fact_ids.issubset(retrieved_ids)
            correctness.append(1.0 if correct else 0.0)
        
        accuracy = np.mean(correctness) if correctness else 0.0
        
        return {
            'accuracy': accuracy,
            'correctness': correctness,
            'memory_used': len(self.memory_store.facts) / self.memory_store.capacity
        }


class HeuristicBaseline:
    """Filter out distractors - keep identifiers, attributes, events."""
    
    def __init__(self, memory_capacity: int = 5, retrieval_budget: int = 3):
        self.memory_store = MemoryStore(capacity=memory_capacity)
        self.retrieval = KeywordRetrieval(top_k=retrieval_budget)
    
    def process_fact(self, fact: Fact) -> bool:
        """Store if not a distractor."""
        if fact.fact_type != 'distractor':
            self.memory_store.write(fact)
            return True
        return False
    
    def answer_question(self, question: Question) -> Tuple[str, List[Fact]]:
        """Retrieve and answer question."""
        retrieved = self.retrieval.retrieve(self.memory_store.facts, question.question_text)
        answer = " ".join([f.content for f in retrieved])
        return answer, retrieved
    
    def process_episode(self, episode: Episode) -> dict:
        """Process episode and track accuracy."""
        self.memory_store.clear()
        
        correctness = []
        
        # Process facts
        for fact in episode.facts:
            self.process_fact(fact)
        
        # Answer questions
        for question in episode.questions:
            answer, retrieved = self.answer_question(question)
            
            required_fact_ids = set(question.required_fact_ids) if question.required_fact_ids else set()
            retrieved_ids = set(f.fact_id for f in retrieved)
            correct = required_fact_ids.issubset(retrieved_ids)
            correctness.append(1.0 if correct else 0.0)
        
        accuracy = np.mean(correctness) if correctness else 0.0
        
        return {
            'accuracy': accuracy,
            'correctness': correctness,
            'memory_used': len(self.memory_store.facts) / self.memory_store.capacity
        }
