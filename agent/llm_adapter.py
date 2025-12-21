"""
Abstract LLM interface for agent reasoning.

This module provides a model-agnostic interface for reasoning oracles,
allowing agents to swap backends without architectural changes.
Currently supports: OpenAI (API), Anthropic (API), LocalMock (testing).

Design principle: The LLM is a black-box solver called only at query time.
All learning, memory selection, and policy updates occur outside the model.
This isolation ensures the contribution is about the agent's decision-making,
not about prompt engineering or model selection.

Backends are interchangeable via factory pattern (get_llm_adapter).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import time
from functools import lru_cache

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv is optional; define a no-op if not installed
    def load_dotenv(*args, **kwargs):
        pass


# Global rate limiter state
_last_call_time = 0.0
_call_count = 0
MIN_SECONDS_BETWEEN_CALLS = 3.0  # Very conservative: 3 seconds minimum between any API calls  # Enforces <= 40 RPM


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    
    Each adapter implements a unified interface for LLM inference,
    allowing the agent to work with different backends transparently.
    
    Built-in features:
    - Rate limiting (enforces min time between calls)
    - Prompt caching (LRU cache for repeated prompts)
    - Call counting (tracks total API calls)
    """
    
    def __init__(self):
        """Initialize adapter with rate limiting."""
        global _call_count
        _call_count = 0
        self._prompt_cache = {}
        self._cache_hits = 0
    
    def _enforce_rate_limit(self):
        """Enforce minimum time between API calls."""
        global _last_call_time
        now = time.time()
        if _last_call_time > 0:
            elapsed = now - _last_call_time
            if elapsed < MIN_SECONDS_BETWEEN_CALLS:
                wait_time = MIN_SECONDS_BETWEEN_CALLS - elapsed
                print(f"      [Rate Limiter] Sleeping {wait_time:.2f}s (elapsed: {elapsed:.2f}s < min: {MIN_SECONDS_BETWEEN_CALLS}s)")
                time.sleep(wait_time)
        _last_call_time = time.time()
    
    def _increment_call_count(self):
        """Track total API calls."""
        global _call_count
        _call_count += 1
    
    def _get_cached_or_generate(self, prompt_key: str, generate_fn):
        """
        Check cache before making API call.
        
        Args:
            prompt_key: Hash or identifier for prompt
            generate_fn: Function to call if cache miss
            
        Returns:
            Cached or generated response
        """
        if prompt_key in self._prompt_cache:
            self._cache_hits += 1
            return self._prompt_cache[prompt_key]
        
        result = generate_fn()
        self._prompt_cache[prompt_key] = result
        return result
    
    def get_stats(self) -> Dict:
        """Get adapter statistics."""
        return {
            'total_calls': _call_count,
            'cache_hits': self._cache_hits,
            'cache_size': len(self._prompt_cache)
        }
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum response length
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def answer_question(self, question: str, context: str = None) -> str:
        """
        Answer a specific question, optionally with context.
        
        Args:
            question: The question to answer
            context: Optional context facts to use
            
        Returns:
            Answer text
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """
    Reasoning oracle backend using OpenAI API.
    
    Implements minimal, stateless inference: accepts prompt, returns text.
    No system messages, prompt engineering, or special handling.
    Used as reference implementation; swap for Anthropic, local models, etc.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (default: gpt-3.5-turbo)
        """
        super().__init__()
        
        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.7) -> str:
        """
        Generate text from prompt. Stateless query to API.
        
        Args:
            prompt: Input text
            max_tokens: Max output length
            temperature: Sampling temperature
            
        Returns:
            Model response (plain text)
        """
        import requests
        
        # Enforce rate limit to prevent quota exhaustion
        self._enforce_rate_limit()
        self._increment_call_count()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise RuntimeError("Invalid API key")
            elif response.status_code == 429:
                raise RuntimeError("Rate limit exceeded. Use --llm mock or wait.")
            raise RuntimeError(f"API error ({response.status_code})")
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {type(e).__name__}")
    
    def answer_question(self, question: str, context: str = None) -> str:
        """
        Answer a question using retrieved context.
        
        Prompt is MINIMIZED to reduce token usage and stay under TPM limits.
        Uses simple format: "Q: X\nFacts:\n- Y\nAnswer:"
        
        Args:
            question: The question
            context: Retrieved context facts
            
        Returns:
            Answer text (cached if same prompt seen before)
        """
        # Minimal prompt format to save tokens
        if context:
            prompt = f"Q: {question}\nFacts:\n{context}\nAnswer:"
        else:
            prompt = f"Q: {question}\nAnswer:"
        
        # Create cache key (hash of prompt)
        cache_key = hash(prompt)
        
        def _generate():
            return self.generate(prompt, max_tokens=30)
        
        # Check cache before API call
        return self._get_cached_or_generate(cache_key, _generate)


class AnthropicAdapter(LLMAdapter):
    """
    Reasoning oracle backend using Anthropic API.
    
    Identical interface to OpenAIAdapter; swappable at runtime.
    """
    
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name
        """
        super().__init__()
        
        load_dotenv()
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.7) -> str:
        """
        Generate text from prompt. Stateless query to API.
        
        Args:
            prompt: Input text
            max_tokens: Max output length
            temperature: Sampling temperature
            
        Returns:
            Model response (plain text)
        """
        import requests
        
        # Enforce rate limit to prevent quota exhaustion
        self._enforce_rate_limit()
        self._increment_call_count()
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2024-06-01",  # Updated to current version
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()['content'][0]['text'].strip()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise RuntimeError("Rate limit exceeded. Use --llm mock or wait.")
            elif response.status_code == 400:
                # Bad request - print actual error for debugging
                try:
                    error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                    raise RuntimeError(f"API error ({response.status_code}): {error_msg}")
                except:
                    raise RuntimeError(f"API error ({response.status_code}): {response.text[:200]}")
            raise RuntimeError(f"API error ({response.status_code})")
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {type(e).__name__}")
    
    def answer_question(self, question: str, context: str = None) -> str:
        """
        Answer a question using retrieved context.
        
        Prompt is MINIMIZED to reduce token usage and stay under TPM limits.
        Uses simple format: "Q: X\nFacts:\n- Y\nAnswer:"
        
        Args:
            question: The question
            context: Retrieved context facts
            
        Returns:
            Answer text (cached if same prompt seen before)
        """
        # Minimal prompt format to save tokens
        if context:
            prompt = f"Q: {question}\nFacts:\n{context}\nAnswer:"
        else:
            prompt = f"Q: {question}\nAnswer:"
        
        # Create cache key (hash of prompt)
        cache_key = hash(prompt)
        
        def _generate():
            return self.generate(prompt, max_tokens=30)
        
        # Check cache before API call
        return self._get_cached_or_generate(cache_key, _generate)


class LocalMockAdapter(LLMAdapter):
    """
    Mock adapter for testing without API calls.
    
    Returns hardcoded responses based on fact content.
    Useful for testing and development.
    """
    
    def __init__(self):
        """Initialize mock adapter."""
        super().__init__()
        self.call_count = 0
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.7) -> str:
        """
        Generate mock response (instant, no API call).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Mock response
        """
        self.call_count += 1
        # Extract answer from prompt if it's a question
        if "A:" in prompt or "Answer:" in prompt:
            return "mock answer"
        return "This is a mock response from the local adapter."
    
    def answer_question(self, question: str, context: str = None) -> str:
        """
        Answer a question with mock response.
        
        Extracts meaningful words from context facts to simulate LLM answer.
        For synthetic QA environment, returns key words from facts.
        
        Args:
            question: The question
            context: Retrieved context facts (newline-separated)
            
        Returns:
            Mock answer word(s) extracted from context
        """
        self.call_count += 1
        
        if not context:
            return "unknown"
        
        # Extract meaningful words from facts
        # Context format: "Fact1\nFact2\n..." where each fact is like "The X is Y." or "The X Y."
        facts = context.split('\n')
        meaningful_words = []
        
        for fact in facts:
            fact = fact.strip()
            if not fact:
                continue
            
            words = [w.rstrip('.,!?;:') for w in fact.split() if w.strip()]
            if not words:
                continue
            
            # Common words to skip
            stop_words = {'the', 'a', 'an', 'is', 'and', 'or', 'in', 'on', 'at', 'of', 'by', 'this', 'that', 'what'}
            meaningful = [w.lower() for w in words if w.lower() not in stop_words]
            
            meaningful_words.extend(meaningful)
        
        # Return all meaningful words found, separated by space
        # This lets similarity_score find the best match
        if meaningful_words:
            return ' '.join(meaningful_words)
        
        return "unknown"


class OllamaAdapter(LLMAdapter):
    """
    Local reasoning oracle using Ollama (llama.cpp backend).
    
    Runs open-source models locally on Mac: no rate limits, fully reproducible.
    Requires: ollama installed and running (ollama serve)
    
    Setup:
        # Install Ollama from https://ollama.ai
        # Start Ollama: ollama serve
        # Pull model: ollama pull mistral (one-time, ~4GB)
    """
    
    def __init__(self, model: str = "deepseek-r1:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama adapter.
        
        Args:
            model: Model name as registered with Ollama (default: deepseek-r1:8b)
            base_url: Ollama server URL (default: localhost:11434)
        """
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama server not responding at {base_url}. "
                    "Start it with: ollama serve"
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                "Install: https://ollama.ai, then run: ollama serve"
            ) from e
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.7) -> str:
        """
        Generate text using local Ollama model (no rate limits).
        
        Args:
            prompt: Input text
            max_tokens: Max output length
            temperature: Sampling temperature
            
        Returns:
            Model response (plain text)
        """
        import requests
        
        # Local models don't need rate limiting, but we track calls
        self._increment_call_count()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timeout (model still loading or processing)")
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Run: ollama serve"
            )
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama error: {str(e)}")
    
    def answer_question(self, question: str, context: str = None) -> str:
        """
        Answer a question using local Ollama model.
        
        Prompt is MINIMIZED to reduce token usage and latency.
        Uses simple format: "Q: X\nFacts:\n- Y\nAnswer:"
        
        Args:
            question: The question
            context: Retrieved context facts
            
        Returns:
            Answer text (cached if same prompt seen before)
        """
        # Minimal prompt format
        if context:
            prompt = f"Q: {question}\nFacts:\n{context}\nAnswer:"
        else:
            prompt = f"Q: {question}\nAnswer:"
        
        # Use cache key
        cache_key = hash(prompt)
        
        def _generate():
            return self.generate(prompt, max_tokens=30)
        
        # Check cache before inference
        return self._get_cached_or_generate(cache_key, _generate)


def get_llm_adapter(adapter_type: str = "openai", **kwargs) -> LLMAdapter:
    """
    Factory function to instantiate reasoning oracle backend.
    
    Design: Backends are interchangeable. Swap OpenAI for Anthropic, local models,
    or other APIs without changing agent code.
    
    Supported backends:
    - "mock": Instant testing (hardcoded responses)
    - "openai": OpenAI API (requires OPENAI_API_KEY)
    - "anthropic": Anthropic API (requires ANTHROPIC_API_KEY)
    - "ollama": Local Ollama model (requires ollama serve running)
    
    Args:
        adapter_type: Backend identifier
        **kwargs: Backend-specific kwargs (api_key, model, base_url, etc.)
        
    Returns:
        LLMAdapter instance
        
    Raises:
        ValueError: If adapter_type unknown
    """
    adapters = {
        'openai': OpenAIAdapter,
        'anthropic': AnthropicAdapter,
        'mock': LocalMockAdapter,
        'ollama': OllamaAdapter
    }
    
    if adapter_type not in adapters:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from {list(adapters.keys())}")
    
    adapter_class = adapters[adapter_type]
    return adapter_class(**kwargs)
