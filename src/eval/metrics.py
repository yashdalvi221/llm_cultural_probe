from typing import Dict, List, Tuple
import math
import random


def accuracy(gold: List[str], pred: List[str]) -> float:
    assert len(gold) == len(pred)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / max(1, len(gold))


def bootstrap_ci(values: List[float], samples: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means: List[float] = []
    for _ in range(samples):
        draw = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lower_idx = int((alpha / 2) * samples)
    upper_idx = int((1 - alpha / 2) * samples)
    return (means[lower_idx], means[min(upper_idx, samples - 1)])


def parity_gap(metric_by_group: Dict[str, float]) -> float:
    if not metric_by_group:
        return 0.0
    vals = list(metric_by_group.values())
    return max(vals) - min(vals)

