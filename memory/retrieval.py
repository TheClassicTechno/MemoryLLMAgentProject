"""
Retrieval methods for facts from memory.
"""
from typing import List, Optional
from env.streaming_qa_env import Fact


class KeywordRetrieval:
    """
    Keyword-based retrieval method for facts.
    
    Retrieves facts matching keywords from queries using simple text matching.
    """
    
    def __init__(self, top_k: int = 3):
        """
        Initialize keyword retrieval.
        
        Args:
            top_k: Maximum number of facts to retrieve
        """
        self.top_k = top_k
    
    def retrieve(self, facts: List[Fact], query: str) -> List[Fact]:
        """
        Retrieve facts matching a query.
        
        Args:
            facts: Pool of facts to search from
            query: Query text to match against facts
            
        Returns:
            List of up to top_k facts, ranked by relevance
        """
        if not facts:
            return []
        
        # Lowercase for comparison
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score each fact based on keyword overlap
        scored_facts = []
        for fact in facts:
            fact_words = set(fact.content.lower().split())
            # Count matching words
            overlap = len(query_words & fact_words)
            if overlap > 0:
                scored_facts.append((fact, overlap))
        
        # Sort by overlap (descending) and return top-k
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        retrieved = [fact for fact, score in scored_facts[:self.top_k]]
        
        return retrieved
    
    def retrieve_exact_ids(self, facts: List[Fact], fact_ids: List[int]) -> List[Fact]:
        """
        Retrieve facts by exact ID match.
        
        Args:
            facts: Pool of facts to search from
            fact_ids: IDs of facts to retrieve
            
        Returns:
            List of facts with matching IDs
        """
        id_set = set(fact_ids)
        return [f for f in facts if f.fact_id in id_set]


class EmbeddingRetrieval:
    """
    Embedding-based retrieval using simple heuristics.
    
    This is a lightweight alternative to full vector similarity search.
    """
    
    def __init__(self, top_k: int = 3):
        """
        Initialize embedding retrieval.
        
        Args:
            top_k: Maximum number of facts to retrieve
        """
        self.top_k = top_k
    
    def _get_fact_type_embedding(self, fact: Fact) -> int:
        """
        Get a simple embedding based on fact type.
        
        Args:
            fact: The fact to embed
            
        Returns:
            Integer embedding code
        """
        type_map = {
            'identifier': 0,
            'attribute': 1,
            'event': 2,
            'distractor': 3
        }
        return type_map.get(fact.fact_type, 3)
    
    def retrieve(self, facts: List[Fact], query_type: str) -> List[Fact]:
        """
        Retrieve facts matching a query type.
        
        Args:
            facts: Pool of facts to search from
            query_type: Type of fact to prioritize
            
        Returns:
            List of up to top_k facts
        """
        if not facts:
            return []
        
        # Prioritize facts matching query type
        typed_facts = []
        for fact in facts:
            if fact.fact_type == query_type:
                typed_facts.append((fact, 0))
            else:
                typed_facts.append((fact, 1))
        
        # Sort by type match, then maintain order
        typed_facts.sort(key=lambda x: x[1])
        retrieved = [fact for fact, _ in typed_facts[:self.top_k]]
        
        return retrieved
