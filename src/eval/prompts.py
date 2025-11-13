from typing import Dict, List


def build_mcq_prompt(stem: str, options: List[str]) -> str:
    labeled = [f"{chr(ord('A')+i)}) {opt}" for i, opt in enumerate(options)]
    return f"{stem}\n\nOptions:\n" + "\n".join(labeled)


