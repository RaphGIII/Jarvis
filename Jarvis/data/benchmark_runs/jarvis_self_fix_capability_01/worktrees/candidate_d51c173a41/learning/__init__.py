"""Developmental learning architecture for JARVIS.

The package keeps parameter learning, memory/statistics, rewards, policies,
world models, curricula, and metacognition separate from the foundation LLM.
"""

from learning.config import LearningConfig

__all__ = ["LearningConfig"]
