from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GenerationRequest:
    prompt: str
    choices: Optional[List[str]] = None  # MCQ if provided
    temperature: float = 0.2
    max_tokens: int = 256
    seed: Optional[int] = 7
    system_prompt: Optional[str] = None
    stop: Optional[List[str]] = None


class ModelClient:
    name: str
    model_id: str

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id

    def generate(self, req: GenerationRequest) -> str:
        raise NotImplementedError("generate must be implemented by subclasses")

    def choose(self, req: GenerationRequest) -> Dict[str, float]:
        """
        Returns a mapping option -> probability. Implementations may return a
        degenerate distribution (argmax 1.0, others 0.0) where token logprobs
        are unavailable.
        """
        raise NotImplementedError("choose must be implemented by subclasses")

