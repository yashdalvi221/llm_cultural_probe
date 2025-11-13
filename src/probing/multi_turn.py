from typing import List, Optional
from ..providers.base import ModelClient, GenerationRequest


def run_multi_turn(
    client: ModelClient,
    context: str,
    probes: List[str],
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> List[str]:
    """
    Simple multi-turn: prepend cultural context as system prompt to each probe.
    """
    outputs: List[str] = []
    for p in probes:
        req = GenerationRequest(
            system_prompt=context,
            prompt=p,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs.append(client.generate(req))
    return outputs

