"""
AI Agent interface for the AutoResearch loop.

The agent reads the current pipeline state, proposes a modification,
and the loop tests whether it improves the metric.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class Proposal:
    """A proposed modification to the prediction pipeline."""
    description: str
    rationale: str
    modification: dict  # e.g., {"model": "XGBoost", "params": {"n_estimators": 200}}

class Agent(ABC):
    """Base class for AutoResearch agents."""
    
    @abstractmethod
    def propose(self, state: dict) -> Proposal:
        """Given current state, propose a pipeline modification."""
        ...

class RandomSearchAgent(Agent):
    """Simple agent that tries random hyperparameter variations."""
    
    def propose(self, state: dict) -> Proposal:
        import random
        # Random hyperparameter suggestion
        n_estimators = random.choice([50, 100, 200, 500])
        max_depth = random.choice([3, 5, 10, None])
        return Proposal(
            description=f"Try n_estimators={n_estimators}, max_depth={max_depth}",
            rationale="Random search exploration",
            modification={"params": {"n_estimators": n_estimators, "max_depth": max_depth}},
        )

class LLMAgent(Agent):
    """Agent that uses an LLM to propose improvements (requires API key)."""
    
    def propose(self, state: dict) -> Proposal:
        # TODO: Call LLM API to propose improvements
        raise NotImplementedError("LLM agent requires groq or openai API key. Install with: pip install orcetra[llm]")