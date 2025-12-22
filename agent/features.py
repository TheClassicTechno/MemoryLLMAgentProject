"""
Feature extraction for memory selection decisions.
"""
import re
from env.streaming_qa_env import Fact


class FeatureExtractor:
    """
    Extracts features from facts for the memory selection policy.
    """
    
    def extract_features(self, fact: Fact, context: dict = None) -> list:
        """
        Extract features from a fact.
        
        Args:
            fact: The fact to extract features from
            context: Optional context information (e.g., time since last fact)
            
        Returns:
            List of feature values
        """
        features = []
        
        # Feature 1: Has numeric content
        has_numbers = bool(re.search(r'\d+', fact.content))
        features.append(1.0 if has_numbers else 0.0)
        
        # Feature 2: Fact type encoding
        type_encoding = {
            'identifier': 1.0,
            'attribute': 0.5,
            'event': 0.5,
            'distractor': 0.0
        }
        features.append(type_encoding.get(fact.fact_type, 0.0))
        
        # Feature 3: Content length (normalized)
        content_length = len(fact.content.split())
        features.append(min(content_length / 20.0, 1.0))
        
        # Feature 4: Has uppercase letters (proper nouns)
        has_uppercase = any(c.isupper() for c in fact.content)
        features.append(1.0 if has_uppercase else 0.0)
        
        # Feature 5: Is distractor
        is_distractor = 1.0 if fact.fact_type == 'distractor' else 0.0
        features.append(is_distractor)
        
        # Feature 6: Contains keywords associated with importance
        important_keywords = {'token', 'key', 'secret', 'password', 'id', 'code', 'identifier'}
        has_important_keyword = any(kw in fact.content.lower() for kw in important_keywords)
        features.append(1.0 if has_important_keyword else 0.0)
        
        return features
    
    def get_feature_names(self) -> list:
        """Get names of extracted features."""
        return [
            'has_numbers',
            'fact_type_score',
            'content_length',
            'has_uppercase',
            'is_distractor',
            'has_important_keyword'
        ]
    
    def feature_dimension(self) -> int:
        """Get dimensionality of feature vectors."""
        return 6
