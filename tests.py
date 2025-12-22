"""
Unit tests for the memory learning project.
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from env.streaming_qa_env import StreamingQAEnvironment, Fact, Question, Episode
from memory.memory_store import MemoryStore
from memory.retrieval import KeywordRetrieval
from agent.features import FeatureExtractor
from agent.policy import MemorySelectionPolicy, RandomPolicy, HeuristicPolicy, NoMemoryPolicy
from agent.core_agent import MemoryAgent
from eval.metrics import MetricsAggregator, TaskMetrics


class TestEnvironment(unittest.TestCase):
    """Tests for streaming QA environment."""
    
    def setUp(self):
        self.env = StreamingQAEnvironment(num_facts=5, num_questions=2)
    
    def test_episode_generation(self):
        """Test that episodes are generated correctly."""
        episode = self.env.generate_episode(0)
        
        self.assertIsNotNone(episode)
        self.assertEqual(len(episode.facts), 5)
        self.assertEqual(len(episode.questions), 2)
        self.assertGreater(episode.delay_length, 0)
    
    def test_fact_generation(self):
        """Test fact generation and properties."""
        episode = self.env.generate_episode(0)
        
        for fact in episode.facts:
            self.assertIsNotNone(fact.content)
            self.assertIn(fact.fact_type, ['identifier', 'attribute', 'event', 'distractor'])
            self.assertGreaterEqual(fact.timestamp, 0)
    
    def test_question_generation(self):
        """Test question generation."""
        episode = self.env.generate_episode(0)
        
        for question in episode.questions:
            self.assertIsNotNone(question.question_text)
            self.assertIsNotNone(question.answer_text)
            self.assertIsInstance(question.required_fact_ids, list)


class TestMemory(unittest.TestCase):
    """Tests for memory storage."""
    
    def setUp(self):
        self.memory = MemoryStore(capacity=3)
    
    def test_write_and_retrieve(self):
        """Test writing and retrieving facts."""
        fact1 = Fact(1, "Test fact 1", 0, "identifier")
        fact2 = Fact(2, "Test fact 2", 1, "attribute")
        
        self.memory.write(fact1)
        self.memory.write(fact2)
        
        retrieved = self.memory.retrieve_all()
        self.assertEqual(len(retrieved), 2)
    
    def test_capacity_limit(self):
        """Test that memory respects capacity limit."""
        for i in range(5):
            fact = Fact(i, f"Fact {i}", i, "distractor")
            self.memory.write(fact)
        
        self.assertEqual(self.memory.get_size(), 3)
    
    def test_fifo_eviction(self):
        """Test FIFO eviction policy."""
        fact1 = Fact(1, "First", 0, "identifier")
        fact2 = Fact(2, "Second", 1, "attribute")
        fact3 = Fact(3, "Third", 2, "event")
        fact4 = Fact(4, "Fourth", 3, "distractor")
        
        self.memory.write(fact1)
        self.memory.write(fact2)
        self.memory.write(fact3)
        self.memory.write(fact4)
        
        retrieved = self.memory.retrieve_all()
        fact_ids = [f.fact_id for f in retrieved]
        self.assertNotIn(1, fact_ids)
        self.assertIn(4, fact_ids)


class TestRetrieval(unittest.TestCase):
    """Tests for retrieval methods."""
    
    def setUp(self):
        self.retrieval = KeywordRetrieval(top_k=2)
        self.facts = [
            Fact(1, "The reset token is orchid", 0, "identifier"),
            Fact(2, "The color is blue", 1, "attribute"),
            Fact(3, "The operation completed", 2, "event"),
            Fact(4, "The status is active", 3, "attribute")
        ]
    
    def test_keyword_retrieval(self):
        """Test keyword-based retrieval."""
        query = "What is the token?"
        retrieved = self.retrieval.retrieve(self.facts, query)
        
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertLessEqual(len(retrieved), 2)
    
    def test_exact_id_retrieval(self):
        """Test exact ID retrieval."""
        retrieved = self.retrieval.retrieve_exact_ids(self.facts, [1, 3])
        
        self.assertEqual(len(retrieved), 2)
        retrieved_ids = {f.fact_id for f in retrieved}
        self.assertEqual(retrieved_ids, {1, 3})


class TestFeatures(unittest.TestCase):
    """Tests for feature extraction."""
    
    def setUp(self):
        self.extractor = FeatureExtractor()
    
    def test_feature_extraction(self):
        """Test feature extraction from facts."""
        fact = Fact(1, "The token is 12345", 0, "identifier")
        features = self.extractor.extract_features(fact)
        
        self.assertEqual(len(features), self.extractor.feature_dimension())
        self.assertTrue(all(0 <= f <= 1 for f in features))
    
    def test_feature_names(self):
        """Test that feature names are available."""
        names = self.extractor.get_feature_names()
        self.assertEqual(len(names), self.extractor.feature_dimension())


class TestPolicies(unittest.TestCase):
    """Tests for memory selection policies."""
    
    def test_learned_policy_decision(self):
        """Test that learned policy makes decisions."""
        policy = MemorySelectionPolicy()
        fact = Fact(1, "Test fact", 0, "identifier")
        
        decision = policy.decide(fact)
        self.assertIsInstance(decision, bool)
    
    def test_random_policy(self):
        """Test random policy."""
        policy = RandomPolicy(write_probability=0.5)
        fact = Fact(1, "Test fact", 0, "identifier")
        
        decisions = [policy.decide(fact) for _ in range(100)]
        true_count = sum(decisions)
        
        # Should be roughly 50% true with 100 trials
        self.assertGreater(true_count, 20)
        self.assertLess(true_count, 80)
    
    def test_heuristic_policy(self):
        """Test heuristic policy."""
        policy = HeuristicPolicy()
        
        distractor = Fact(1, "Irrelevant fact", 0, "distractor")
        identifier = Fact(2, "The token is X", 1, "identifier")
        
        self.assertFalse(policy.decide(distractor))
        self.assertTrue(policy.decide(identifier))
    
    def test_no_memory_policy(self):
        """Test that no-memory policy never stores."""
        policy = NoMemoryPolicy()
        fact = Fact(1, "Test fact", 0, "identifier")
        
        self.assertFalse(policy.decide(fact))


class TestAgent(unittest.TestCase):
    """Tests for core agent."""
    
    def setUp(self):
        self.agent = MemoryAgent(memory_capacity=3, retrieval_budget=2)
    
    def test_fact_processing(self):
        """Test that agent processes facts."""
        fact = Fact(1, "Test fact", 0, "identifier")
        decision = self.agent.process_fact(fact)
        
        self.assertIsInstance(decision, bool)
    
    def test_episode_processing(self):
        """Test complete episode processing."""
        env = StreamingQAEnvironment(num_facts=5, num_questions=2)
        episode = env.generate_episode(0)
        
        metrics = self.agent.process_episode(episode)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('correct_answers', metrics)
        self.assertIn('total_questions', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)


class TestMetrics(unittest.TestCase):
    """Tests for metrics and evaluation."""
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation."""
        aggregator = MetricsAggregator()
        
        metrics1 = {
            'episode_id': 0,
            'accuracy': 0.8,
            'correct_answers': 2,
            'total_questions': 2,
            'memory_used': 3,
            'memory_capacity': 5,
            'write_count': 5,
            'memory_stats': {},
            'question_results': []
        }
        
        aggregator.add_episode(metrics1)
        summary = aggregator.get_summary()
        
        self.assertEqual(summary['num_episodes'], 1)
        self.assertEqual(summary['mean_accuracy'], 0.8)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrieval))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicies))
    suite.addTests(loader.loadTestsFromTestCase(TestAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
