"""
Streaming QA environment for agent memory learning.

This module provides a synthetic task where an agent observes
facts over time and must answer questions based on those facts.
"""
import random
import json
from typing import List, Dict, Tuple, Optional


class Fact:
    """Represents a single factual observation."""
    
    def __init__(self, fact_id: int, content: str, timestamp: int, fact_type: str):
        """
        Initialize a fact.
        
        Args:
            fact_id: Unique identifier for the fact
            content: The actual content/text of the fact
            timestamp: When the fact was observed
            fact_type: Category of the fact (e.g., 'identifier', 'attribute', 'event')
        """
        self.fact_id = fact_id
        self.content = content
        self.timestamp = timestamp
        self.fact_type = fact_type
    
    def to_dict(self) -> Dict:
        """Convert fact to dictionary representation."""
        return {
            'fact_id': self.fact_id,
            'content': self.content,
            'timestamp': self.timestamp,
            'fact_type': self.fact_type
        }


class Question:
    """Represents a question that requires factual recall."""
    
    def __init__(self, question_id: int, question_text: str, 
                 required_fact_ids: List[int], answer_text: str):
        """
        Initialize a question.
        
        Args:
            question_id: Unique identifier for the question
            question_text: The actual question text
            required_fact_ids: List of fact IDs needed to answer correctly
            answer_text: The correct answer text
        """
        self.question_id = question_id
        self.question_text = question_text
        self.required_fact_ids = required_fact_ids
        self.answer_text = answer_text
    
    def to_dict(self) -> Dict:
        """Convert question to dictionary representation."""
        return {
            'question_id': self.question_id,
            'question_text': self.question_text,
            'required_fact_ids': self.required_fact_ids,
            'answer_text': self.answer_text
        }


class Episode:
    """Represents a single episode of the streaming QA task."""
    
    def __init__(self, episode_id: int, facts: List[Fact], 
                 questions: List[Question], delay_length: int):
        """
        Initialize an episode.
        
        Args:
            episode_id: Unique identifier for the episode
            facts: List of factual observations
            questions: List of questions to answer
            delay_length: Number of time steps before questions are asked
        """
        self.episode_id = episode_id
        self.facts = facts
        self.questions = questions
        self.delay_length = delay_length
    
    def to_dict(self) -> Dict:
        """Convert episode to dictionary representation."""
        return {
            'episode_id': self.episode_id,
            'facts': [f.to_dict() for f in self.facts],
            'questions': [q.to_dict() for q in self.questions],
            'delay_length': self.delay_length
        }


class StreamingQAEnvironment:
    """
    Streaming QA environment for memory learning.
    
    The agent observes facts over time, must decide which to remember,
    and then answers questions based on retrieved memories.
    """
    
    def __init__(self, num_facts: int = 10, num_questions: int = 2, 
                 delay_length: int = 3, distractor_density: float = 0.3):
        """
        Initialize the environment.
        
        Args:
            num_facts: Number of facts per episode
            num_questions: Number of questions per episode
            delay_length: Time steps between facts and questions
            distractor_density: Proportion of distractor facts (0.0 to 1.0)
        """
        self.num_facts = num_facts
        self.num_questions = num_questions
        self.delay_length = delay_length
        self.distractor_density = distractor_density
        
        self.current_episode: Optional[Episode] = None
        self.fact_counter = 0
        self.question_counter = 0
    
    def _generate_fact(self, fact_index: int) -> Fact:
        """Generate a single fact."""
        is_distractor = random.random() < self.distractor_density
        
        if is_distractor:
            fact_type = 'distractor'
            content = f"Distraction fact: {random.choice(['The sky is blue', 'Water is wet', 'Grass is green', 'Fire is hot', 'Ice is cold'])}"
        else:
            fact_type = random.choice(['identifier', 'attribute', 'event'])
            
            if fact_type == 'identifier':
                identifiers = ['token', 'id', 'code', 'key', 'secret', 'password']
                values = ['orchid', 'amber', 'silver', 'golden', 'crimson', 'azure']
                content = f"The {random.choice(identifiers)} is {random.choice(values)}."
            elif fact_type == 'attribute':
                attributes = ['color', 'size', 'status', 'level', 'count', 'type']
                values = ['red', 'large', 'active', 'high', 'five', 'primary']
                content = f"The {random.choice(attributes)} is {random.choice(values)}."
            else:  # event
                events = ['started', 'completed', 'initiated', 'triggered', 'executed', 'processed']
                items = ['task', 'operation', 'procedure', 'transaction', 'query', 'request']
                content = f"The {random.choice(items)} {random.choice(events)}."
        
        return Fact(
            fact_id=self.fact_counter,
            content=content,
            timestamp=fact_index,
            fact_type=fact_type
        )
    
    def _generate_question(self, relevant_facts: List[Fact]) -> Question:
        """Generate a question based on relevant facts."""
        if not relevant_facts:
            # Fallback if no relevant facts
            return Question(
                question_id=self.question_counter,
                question_text="What is a color?",
                required_fact_ids=[],
                answer_text="A color is a property of light."
            )
        
        selected_fact = random.choice(relevant_facts)
        required_fact_ids = [selected_fact.fact_id]
        
        if selected_fact.fact_type == 'identifier':
            words = selected_fact.content.split()
            question_text = f"What is the {words[2]}?"
            answer_text = words[-1].rstrip('.')
        elif selected_fact.fact_type == 'attribute':
            words = selected_fact.content.split()
            question_text = f"What is the {words[2]}?"
            answer_text = words[-1].rstrip('.')
        else:  # event
            words = selected_fact.content.split()
            question_text = f"What {words[2].lower()} in this episode?"
            answer_text = words[1].rstrip('.')
        
        return Question(
            question_id=self.question_counter,
            question_text=question_text,
            required_fact_ids=required_fact_ids,
            answer_text=answer_text
        )
    
    def generate_episode(self, episode_id: int) -> Episode:
        """
        Generate a new episode with facts and questions.
        
        Args:
            episode_id: Identifier for the episode
            
        Returns:
            A new Episode object
        """
        self.fact_counter = 0
        self.question_counter = 0
        
        # Generate facts
        facts = []
        for i in range(self.num_facts):
            fact = self._generate_fact(i)
            facts.append(fact)
            self.fact_counter += 1
        
        # Identify non-distractor facts for question generation
        relevant_facts = [f for f in facts if f.fact_type != 'distractor']
        
        # Generate questions
        questions = []
        for i in range(self.num_questions):
            question = self._generate_question(relevant_facts)
            questions.append(question)
            self.question_counter += 1
        
        episode = Episode(
            episode_id=episode_id,
            facts=facts,
            questions=questions,
            delay_length=self.delay_length
        )
        
        self.current_episode = episode
        return episode
    
    def get_episode_facts(self) -> List[Fact]:
        """Get facts from current episode in order."""
        if not self.current_episode:
            raise RuntimeError("No active episode")
        return self.current_episode.facts
    
    def get_episode_questions(self) -> List[Question]:
        """Get questions from current episode."""
        if not self.current_episode:
            raise RuntimeError("No active episode")
        return self.current_episode.questions
    
    def evaluate_answer(self, question: Question, retrieved_facts: List[Fact], 
                       agent_answer: str) -> bool:
        """
        Evaluate if agent answer is correct.
        
        Args:
            question: The question being answered
            retrieved_facts: Facts the agent retrieved from memory
            agent_answer: The agent's answer text
            
        Returns:
            True if answer is correct, False otherwise
        """
        # Check if all required facts were retrieved
        retrieved_ids = [f.fact_id for f in retrieved_facts]
        has_required_facts = all(fid in retrieved_ids for fid in question.required_fact_ids)
        
        if not has_required_facts:
            return False
        
        # Simple string matching for answer evaluation
        agent_answer_lower = agent_answer.lower().strip()
        correct_answer_lower = question.answer_text.lower().strip()
        
        return agent_answer_lower == correct_answer_lower
