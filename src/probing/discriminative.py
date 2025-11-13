from typing import Dict, List, Tuple
from ..providers.base import ModelClient, GenerationRequest


def run_batch_choose(
    client: ModelClient, stems: List[str], choices: List[List[str]], temperature: float = 0.2
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for stem, opts in zip(stems, choices):
        req = GenerationRequest(prompt=stem, choices=opts, temperature=temperature)
        probs = client.choose(req)
        results.append(probs)
    return results

