"""
AI Client for QA Analysis Insights

This module provides a unified interface for using various AI providers
(Claude, GPT-4, local LLMs) to analyze QA data and provide insights.
"""

import os
import json
import time
import sqlite3
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Generator
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
import pandas as pd

# AI Provider imports
try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests  # For Ollama

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Cost tracking constants (per 1K tokens)
PRICING = {
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "local": {"input": 0.0, "output": 0.0}  # Free for local models
}


class AIProvider(Enum):
    """Supported AI providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class AIResponse:
    """Standardized AI response."""
    content: str
    provider: str
    model: str
    tokens_used: Dict[str, int]
    cost: float
    cached: bool = False
    response_time: float = 0.0


@dataclass
class AnalysisPrompt:
    """Structured prompt for analysis."""
    system: str
    user: str
    context: Optional[Dict[str, Any]] = None
    max_tokens: int = 2000
    temperature: float = 0.7


class PromptTemplates:
    """QA-specific prompt templates."""

    @staticmethod
    def failure_pattern_analysis() -> AnalysisPrompt:
        """Analyze failure patterns in QA data."""
        return AnalysisPrompt(
            system="""You are an expert quality assurance analyst specializing in potentiometer manufacturing.
            Your role is to analyze failure patterns and provide actionable insights.
            Focus on:
            1. Identifying systematic issues vs random failures
            2. Correlating failures with specific parameters
            3. Suggesting root causes
            4. Recommending preventive actions""",
            user="""Analyze the following failure data and provide insights:

{data}

Please identify:
1. Common failure patterns
2. Critical parameters contributing to failures
3. Potential root causes
4. Recommended actions to reduce failure rate""",
            temperature=0.3  # Lower temperature for analytical tasks
        )

    @staticmethod
    def process_improvement() -> AnalysisPrompt:
        """Suggest process improvements based on data."""
        return AnalysisPrompt(
            system="""You are a manufacturing process improvement expert with deep knowledge of potentiometer production.
            Your goal is to identify opportunities for process optimization based on QA data.
            Consider:
            1. Manufacturing efficiency
            2. Quality metrics improvement
            3. Cost reduction opportunities
            4. Equipment and tooling optimization""",
            user="""Based on the following QA data analysis:

{data}

Suggest process improvements for:
1. Reducing sigma gradient variations
2. Improving linearity performance
3. Minimizing resistance drift
4. Enhancing overall yield

Provide specific, actionable recommendations with expected impact.""",
            temperature=0.5
        )

    @staticmethod
    def qa_report_generation() -> AnalysisPrompt:
        """Generate comprehensive QA report."""
        return AnalysisPrompt(
            system="""You are a QA report specialist who creates clear, professional reports for management and engineering teams.
            Your reports should be:
            1. Data-driven and factual
            2. Easy to understand for non-technical stakeholders
            3. Actionable with clear recommendations
            4. Properly structured with executive summary""",
            user="""Generate a comprehensive QA report based on:

{data}

Include:
1. Executive Summary
2. Key Performance Indicators
3. Trend Analysis
4. Risk Assessment
5. Recommendations
6. Action Items with priorities

Format the report professionally with clear sections.""",
            temperature=0.4
        )

    @staticmethod
    def data_interpretation() -> AnalysisPrompt:
        """Help interpret complex QA data."""
        return AnalysisPrompt(
            system="""You are a data interpretation expert for potentiometer QA.
            Help users understand complex metrics and their implications.
            Explain technical concepts in accessible terms while maintaining accuracy.""",
            user="""Please help interpret this QA data:

{data}

Question: {question}

Provide a clear explanation that covers:
1. What the data shows
2. Why it matters
3. Normal vs abnormal ranges
4. Implications for product quality""",
            temperature=0.3
        )


class ResponseCache:
    """Cache AI responses to reduce API calls and costs."""

    def __init__(self, cache_dir: str = "ai_cache", ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        os.makedirs(cache_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize cache database."""
        db_path = os.path.join(self.cache_dir, "response_cache.db")
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                hash TEXT PRIMARY KEY,
                response BLOB,
                created_at TIMESTAMP,
                provider TEXT,
                model TEXT,
                tokens_used TEXT,
                cost REAL
            )
        ''')
        self.conn.commit()

    def _generate_hash(self, prompt: AnalysisPrompt, provider: str, model: str) -> str:
        """Generate unique hash for cache key."""
        content = f"{prompt.system}{prompt.user}{str(prompt.context)}{provider}{model}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: AnalysisPrompt, provider: str, model: str) -> Optional[AIResponse]:
        """Retrieve cached response if available and not expired."""
        hash_key = self._generate_hash(prompt, provider, model)

        cursor = self.conn.execute(
            "SELECT response, created_at, tokens_used, cost FROM cache WHERE hash = ?",
            (hash_key,)
        )
        row = cursor.fetchone()

        if row:
            response_data, created_at, tokens_str, cost = row
            created_time = datetime.fromisoformat(created_at)

            if datetime.now() - created_time < self.ttl:
                response = pickle.loads(response_data)
                response.cached = True
                return response

        return None

    def set(self, prompt: AnalysisPrompt, response: AIResponse):
        """Cache a response."""
        hash_key = self._generate_hash(prompt, response.provider, response.model)

        self.conn.execute('''
            INSERT OR REPLACE INTO cache 
            (hash, response, created_at, provider, model, tokens_used, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            hash_key,
            pickle.dumps(response),
            datetime.now().isoformat(),
            response.provider,
            response.model,
            json.dumps(response.tokens_used),
            response.cost
        ))
        self.conn.commit()

    def clear_expired(self):
        """Remove expired cache entries."""
        expiry_time = datetime.now() - self.ttl
        self.conn.execute(
            "DELETE FROM cache WHERE created_at < ?",
            (expiry_time.isoformat(),)
        )
        self.conn.commit()


class BaseAIClient(ABC):
    """Base class for AI providers."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def complete(self, prompt: AnalysisPrompt) -> AIResponse:
        """Get completion from AI provider."""
        pass

    @abstractmethod
    def stream_complete(self, prompt: AnalysisPrompt) -> Generator[str, None, AIResponse]:
        """Stream completion from AI provider."""
        pass

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost based on token usage."""
        pricing = PRICING.get(self.model, {"input": 0, "output": 0})
        cost = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1000
        return cost


class AnthropicClient(BaseAIClient):
    """Anthropic Claude API client."""

    def __init__(self, model: str = "claude-3-sonnet", api_key: Optional[str] = None):
        super().__init__(model, api_key or os.getenv("ANTHROPIC_API_KEY"))
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def complete(self, prompt: AnalysisPrompt) -> AIResponse:
        """Get completion from Claude."""
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}]
            )

            content = response.content[0].text
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens

            return AIResponse(
                content=content,
                provider=AIProvider.ANTHROPIC.value,
                model=self.model,
                tokens_used={"input": tokens_in, "output": tokens_out},
                cost=self.calculate_cost(tokens_in, tokens_out),
                response_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise

    def stream_complete(self, prompt: AnalysisPrompt) -> Generator[str, None, AIResponse]:
        """Stream completion from Claude."""
        start_time = time.time()
        full_response = ""

        try:
            with self.client.messages.stream(
                    model=self.model,
                    max_tokens=prompt.max_tokens,
                    temperature=prompt.temperature,
                    system=prompt.system,
                    messages=[{"role": "user", "content": prompt.user}]
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield text

                # Get final usage info
                message = stream.get_final_message()
                tokens_in = message.usage.input_tokens
                tokens_out = message.usage.output_tokens

                return AIResponse(
                    content=full_response,
                    provider=AIProvider.ANTHROPIC.value,
                    model=self.model,
                    tokens_used={"input": tokens_in, "output": tokens_out},
                    cost=self.calculate_cost(tokens_in, tokens_out),
                    response_time=time.time() - start_time
                )

        except Exception as e:
            self.logger.error(f"Anthropic streaming error: {str(e)}")
            raise


class OpenAIClient(BaseAIClient):
    """OpenAI GPT API client."""

    def __init__(self, model: str = "gpt-4-turbo", api_key: Optional[str] = None):
        super().__init__(model, api_key or os.getenv("OPENAI_API_KEY"))
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def complete(self, prompt: AnalysisPrompt) -> AIResponse:
        """Get completion from GPT."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user}
                ]
            )

            content = response.choices[0].message.content
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens

            return AIResponse(
                content=content,
                provider=AIProvider.OPENAI.value,
                model=self.model,
                tokens_used={"input": tokens_in, "output": tokens_out},
                cost=self.calculate_cost(tokens_in, tokens_out),
                response_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise

    def stream_complete(self, prompt: AnalysisPrompt) -> Generator[str, None, AIResponse]:
        """Stream completion from GPT."""
        start_time = time.time()
        full_response = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user}
                ],
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            # Estimate tokens for streaming (OpenAI doesn't provide in stream)
            tokens_in = self.count_tokens(prompt.system + prompt.user)
            tokens_out = self.count_tokens(full_response)

            return AIResponse(
                content=full_response,
                provider=AIProvider.OPENAI.value,
                model=self.model,
                tokens_used={"input": tokens_in, "output": tokens_out},
                cost=self.calculate_cost(tokens_in, tokens_out),
                response_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {str(e)}")
            raise


class OllamaClient(BaseAIClient):
    """Ollama local LLM client."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        super().__init__(model)
        if not HAS_OLLAMA:
            raise ImportError("requests package not installed. Run: pip install requests")
        self.base_url = base_url

    def complete(self, prompt: AnalysisPrompt) -> AIResponse:
        """Get completion from Ollama."""
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{prompt.system}\n\n{prompt.user}",
                    "stream": False,
                    "options": {
                        "temperature": prompt.temperature,
                        "num_predict": prompt.max_tokens
                    }
                }
            )
            response.raise_for_status()

            data = response.json()
            content = data["response"]

            # Estimate tokens
            tokens_in = self.count_tokens(prompt.system + prompt.user)
            tokens_out = self.count_tokens(content)

            return AIResponse(
                content=content,
                provider=AIProvider.OLLAMA.value,
                model=self.model,
                tokens_used={"input": tokens_in, "output": tokens_out},
                cost=0.0,  # Local models are free
                response_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            raise

    def stream_complete(self, prompt: AnalysisPrompt) -> Generator[str, None, AIResponse]:
        """Stream completion from Ollama."""
        start_time = time.time()
        full_response = ""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{prompt.system}\n\n{prompt.user}",
                    "stream": True,
                    "options": {
                        "temperature": prompt.temperature,
                        "num_predict": prompt.max_tokens
                    }
                },
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        text = data["response"]
                        full_response += text
                        yield text

            # Estimate tokens
            tokens_in = self.count_tokens(prompt.system + prompt.user)
            tokens_out = self.count_tokens(full_response)

            return AIResponse(
                content=full_response,
                provider=AIProvider.OLLAMA.value,
                model=self.model,
                tokens_used={"input": tokens_in, "output": tokens_out},
                cost=0.0,
                response_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Ollama streaming error: {str(e)}")
            raise


class QAAIAnalyzer:
    """Main AI analyzer for QA insights with retry logic and caching."""

    def __init__(
            self,
            provider: AIProvider = AIProvider.ANTHROPIC,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            cache_ttl_hours: int = 24,
            max_retries: int = 3,
            retry_delay: float = 1.0
    ):
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache = ResponseCache(ttl_hours=cache_ttl_hours)
        self.logger = logging.getLogger(__name__)

        # Initialize the appropriate client
        if provider == AIProvider.ANTHROPIC:
            self.client = AnthropicClient(model or "claude-3-sonnet", api_key)
        elif provider == AIProvider.OPENAI:
            self.client = OpenAIClient(model or "gpt-4-turbo", api_key)
        elif provider == AIProvider.OLLAMA:
            self.client = OllamaClient(model or "llama2")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Track usage for session
        self.session_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_tokens": {"input": 0, "output": 0},
            "total_cost": 0.0,
            "total_time": 0.0
        }

    def analyze_failures(self, data: Union[str, Dict, List]) -> AIResponse:
        """Analyze failure patterns in QA data."""
        prompt = PromptTemplates.failure_pattern_analysis()
        prompt.user = prompt.user.format(data=self._format_data(data))
        return self._execute_with_retry(prompt)

    def suggest_improvements(self, data: Union[str, Dict, List]) -> AIResponse:
        """Suggest process improvements based on QA data."""
        prompt = PromptTemplates.process_improvement()
        prompt.user = prompt.user.format(data=self._format_data(data))
        return self._execute_with_retry(prompt)

    def generate_report(self, data: Union[str, Dict, List]) -> AIResponse:
        """Generate comprehensive QA report."""
        prompt = PromptTemplates.qa_report_generation()
        prompt.user = prompt.user.format(data=self._format_data(data))
        return self._execute_with_retry(prompt)

    def interpret_data(self, data: Union[str, Dict, List], question: str) -> AIResponse:
        """Help interpret QA data based on specific question."""
        prompt = PromptTemplates.data_interpretation()
        prompt.user = prompt.user.format(
            data=self._format_data(data),
            question=question
        )
        return self._execute_with_retry(prompt)

    def custom_analysis(
            self,
            system_prompt: str,
            user_prompt: str,
            data: Optional[Union[str, Dict, List]] = None,
            **kwargs
    ) -> AIResponse:
        """Perform custom analysis with user-defined prompts."""
        if data:
            user_prompt = user_prompt.format(data=self._format_data(data), **kwargs)

        prompt = AnalysisPrompt(
            system=system_prompt,
            user=user_prompt,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.7)
        )

        return self._execute_with_retry(prompt)

    def stream_analysis(
            self,
            prompt_type: str,
            data: Union[str, Dict, List],
            **kwargs
    ) -> Generator[str, None, AIResponse]:
        """Stream analysis results for real-time display."""
        # Get appropriate prompt template
        if prompt_type == "failures":
            prompt = PromptTemplates.failure_pattern_analysis()
        elif prompt_type == "improvements":
            prompt = PromptTemplates.process_improvement()
        elif prompt_type == "report":
            prompt = PromptTemplates.qa_report_generation()
        elif prompt_type == "interpret":
            prompt = PromptTemplates.data_interpretation()
            prompt.user = prompt.user.format(
                data=self._format_data(data),
                question=kwargs.get("question", "")
            )
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        if prompt_type != "interpret":
            prompt.user = prompt.user.format(data=self._format_data(data))

        # Stream the response
        return self._stream_with_retry(prompt)

    def _format_data(self, data: Union[str, Dict, List]) -> str:
        """Format data for prompt inclusion."""
        if isinstance(data, str):
            return data
        elif isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)
        else:
            return str(data)

    def _execute_with_retry(self, prompt: AnalysisPrompt) -> AIResponse:
        """Execute request with retry logic and caching."""
        # Check cache first
        cached_response = self.cache.get(prompt, self.provider.value, self.client.model)
        if cached_response:
            self.session_stats["cache_hits"] += 1
            self.logger.info("Using cached response")
            return cached_response

        # Try to get response with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{self.max_retries}")
                response = self.client.complete(prompt)

                # Update session stats
                self.session_stats["total_requests"] += 1
                self.session_stats["total_tokens"]["input"] += response.tokens_used["input"]
                self.session_stats["total_tokens"]["output"] += response.tokens_used["output"]
                self.session_stats["total_cost"] += response.cost
                self.session_stats["total_time"] += response.response_time

                # Cache the response
                self.cache.set(prompt, response)

                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        raise Exception(f"All retry attempts failed. Last error: {str(last_error)}")

    def _stream_with_retry(self, prompt: AnalysisPrompt) -> Generator[str, None, AIResponse]:
        """Stream request with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Stream attempt {attempt + 1}/{self.max_retries}")

                # Stream the response
                generator = self.client.stream_complete(prompt)

                # Yield chunks
                for chunk in generator:
                    yield chunk

                # Get the final response (generator should return AIResponse)
                try:
                    response = generator.send(None)
                except StopIteration as e:
                    response = e.value

                # Update session stats
                self.session_stats["total_requests"] += 1
                self.session_stats["total_tokens"]["input"] += response.tokens_used["input"]
                self.session_stats["total_tokens"]["output"] += response.tokens_used["output"]
                self.session_stats["total_cost"] += response.cost
                self.session_stats["total_time"] += response.response_time

                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Stream attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise Exception(f"All stream attempts failed. Last error: {str(last_error)}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get usage statistics for current session."""
        return {
            **self.session_stats,
            "cache_hit_rate": (
                                      self.session_stats["cache_hits"] / max(1, self.session_stats["total_requests"])
                              ) * 100,
            "average_response_time": (
                    self.session_stats["total_time"] / max(1, self.session_stats["total_requests"])
            )
        }

    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear_expired()
        self.logger.info("Cache cleared")


# Example usage and integration
def main():
    """Example usage of the QA AI Analyzer."""

    # Sample QA data
    sample_data = {
        "model": "8340-1",
        "total_units": 150,
        "failures": 12,
        "metrics": {
            "avg_sigma_gradient": 0.0234,
            "sigma_threshold": 0.4,
            "linearity_pass_rate": 0.92,
            "resistance_drift": 2.3
        },
        "failure_details": [
            {"serial": "A12345", "sigma": 0.41, "reason": "Exceeded threshold"},
            {"serial": "A12346", "sigma": 0.39, "reason": "Linearity failure"}
        ]
    }

    # Initialize analyzer (defaults to Claude)
    analyzer = QAAIAnalyzer(
        provider=AIProvider.ANTHROPIC,
        model="claude-3-haiku",  # Using cheaper model for example
        cache_ttl_hours=24
    )

    # Example 1: Analyze failures
    print("=== Analyzing Failures ===")
    response = analyzer.analyze_failures(sample_data)
    print(f"Analysis: {response.content[:500]}...")
    print(f"Cost: ${response.cost:.4f}")
    print(f"Response time: {response.response_time:.2f}s")

    # Example 2: Suggest improvements
    print("\n=== Suggesting Improvements ===")
    response = analyzer.suggest_improvements(sample_data)
    print(f"Suggestions: {response.content[:500]}...")

    # Example 3: Generate report
    print("\n=== Generating Report ===")
    response = analyzer.generate_report(sample_data)
    print(f"Report: {response.content[:500]}...")

    # Example 4: Ask specific question
    print("\n=== Data Interpretation ===")
    response = analyzer.interpret_data(
        sample_data,
        "Why is the sigma gradient higher than usual for this batch?"
    )
    print(f"Interpretation: {response.content[:500]}...")

    # Example 5: Streaming response
    print("\n=== Streaming Analysis ===")
    print("Streaming failure analysis: ", end="", flush=True)
    for chunk in analyzer.stream_analysis("failures", sample_data):
        print(chunk, end="", flush=True)
    print()

    # Show session statistics
    print("\n=== Session Statistics ===")
    stats = analyzer.get_session_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1f}%)")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"Average response time: {stats['average_response_time']:.2f}s")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()