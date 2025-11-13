from typing import List
from ..providers.base import ModelClient, GenerationRequest


def run_batch_generate(
    client: ModelClient, prompts: List[str], temperature: float = 0.2, max_tokens: int = 256
) -> List[str]:
    outputs: List[str] = []
    for p in prompts:
        req = GenerationRequest(prompt=p, temperature=temperature, max_tokens=max_tokens)
        outputs.append(client.generate(req))
    return outputs

