"""
Memory storage backend for facts.
"""
from typing import List, Optional
from env.streaming_qa_env import Fact


class MemoryStore:
    """
    Simple memory storage with fixed capacity and FIFO eviction.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the memory store.
        
        Args:
            capacity: Maximum number of facts the memory can store
        """
        self.capacity = capacity
        self.facts: List[Fact] = []
        self.write_count = 0
    
    def write(self, fact: Fact) -> bool:
        """
        Write a fact to memory.
        
        If memory is full, evict the oldest fact (FIFO).
        
        Args:
            fact: The fact to write
            
        Returns:
            True if write was successful, False if memory was full and eviction occurred
        """
        if len(self.facts) >= self.capacity:
            self.facts.pop(0)
            evicted = True
        else:
            evicted = False
        
        self.facts.append(fact)
        self.write_count += 1
        
        return not evicted
    
    def retrieve_all(self) -> List[Fact]:
        """Get all facts currently in memory."""
        return self.facts.copy()
    
    def retrieve_by_ids(self, fact_ids: List[int]) -> List[Fact]:
        """
        Retrieve facts by their IDs.
        
        Args:
            fact_ids: List of fact IDs to retrieve
            
        Returns:
            List of facts matching the IDs
        """
        id_set = set(fact_ids)
        return [f for f in self.facts if f.fact_id in id_set]
    
    def clear(self):
        """Clear all facts from memory."""
        self.facts.clear()
        self.write_count = 0
    
    def get_size(self) -> int:
        """Get current number of facts in memory."""
        return len(self.facts)
    
    def is_full(self) -> bool:
        """Check if memory is at capacity."""
        return len(self.facts) >= self.capacity
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            'current_size': len(self.facts),
            'capacity': self.capacity,
            'write_count': self.write_count,
            'utilization': len(self.facts) / self.capacity
        }
