"""
Adversarial Memory - A unified framework for evaluating LLM memory systems.
"""

__version__ = "0.1.0"

# Core types
from .types import (
    ConversationID,
    LLMResponse,
    Prompt,
    PromptContext,
    Message,
    Conversation,
    LLM,
    MemorySystem,
    EvaluationPromptTemplate,
)

# LLM interface
from .llm import (
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
)

# Memory systems
from .memory import (
    NoHistoryMemorySystem,
    SimpleHistoryMemorySystem,
)
from .memory_mem0 import Mem0MemorySystem

# MemFail's published repo does not include the memory-system adapters the paper
# evaluates. Ones we have not implemented stay None so importing the harness
# still works; selecting them raises a clear error instead of an ImportError.
from .memory_amem import AMEMMemorySystem
EverMemOSMemorySystem = None
LiCoMemoryMemorySystem = None
SimpleMemMemorySystem = None
try:
    from .memory_structmem import StructMemMemorySystem   # needs LightMem (py<3.12)
except Exception:
    StructMemMemorySystem = None
try:
    from .memory_letta import LettaMemorySystem           # needs letta_client + server
except Exception:
    LettaMemorySystem = None

# Chat systems
from .chat import (
    ChatSystem,
)

# Dataset and Evaluation
from .dataset import (
    ChatDataset,
    ConversationData,
)
from .evaluation import (
    Evaluator,
    EvaluationResult,
    EvaluationSummary,
)
from .tokenizer import (
    EvaluationTokenizer,
    TiktokenTokenizer,
)
from .prompt_templates import (
    SimplePromptTemplate,
    ConversationHistoryPromptTemplate,
)

__all__ = [
    # Types
    "ConversationID",
    "LLMResponse",
    "Prompt",
    "PromptContext",
    "Message",
    "Conversation",
    "LLM",
    "MemorySystem",
    "ChatSystem",
    # LLM
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    # Memory
    "NoHistoryMemorySystem",
    "SimpleHistoryMemorySystem",
    "Mem0MemorySystem",
    "AMEMMemorySystem",
    "SimpleMemMemorySystem",
    "EverMemOSMemorySystem",
    "StructMemMemorySystem",
    "LettaMemorySystem",
    "LiCoMemoryMemorySystem",
    # Chat
    "ChatSystem",
    # Dataset
    "ChatDataset",
    "ConversationData",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "EvaluationSummary",
    # Types
    "EvaluationPromptTemplate",
    # Tokenizer
    "EvaluationTokenizer",
    "TiktokenTokenizer",
    # Prompt Templates
    "SimplePromptTemplate",
    "ConversationHistoryPromptTemplate",
]
