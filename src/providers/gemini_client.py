import os
from typing import Dict, List, Optional

from .base import GenerationRequest, ModelClient
from ..utils.cache import ContentHashCache
from ..utils.rate_limit import retry_with_backoff


class GeminiClient(ModelClient):
    def __init__(
        self,
        model_id: str = "gemini-flash-latest",
        name: str = "gemini-flash-latest",
        cache_dir: str = ".cache",
    ) -> None:
        super().__init__(name=name, model_id=model_id)
        self._cache = ContentHashCache(cache_dir=cache_dir)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=model_id)

    def _cache_key(self, kind: str, req: GenerationRequest) -> str:
        payload = {
            "provider": "gemini",
            "kind": kind,
            "model_id": self.model_id,
            "prompt": req.prompt,
            "choices": req.choices,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "system_prompt": req.system_prompt,
            "stop": req.stop,
        }
        return self._cache.key_for(payload)

    @retry_with_backoff(Exception)
    def generate(self, req: GenerationRequest) -> str:
        key = self._cache_key("generate", req)
        cached = self._cache.get(key)
        if cached:
            return cached.get("text", "")

        parts: List[str] = []
        if req.system_prompt:
            parts.append(req.system_prompt)
        parts.append(req.prompt)
        text_input = "\n\n".join(parts)

        resp = self._model.generate_content(
            text_input,
            generation_config={
                "temperature": req.temperature,
                "max_output_tokens": req.max_tokens,
                # stopSequences name in Gemini
                "stop_sequences": req.stop or [],
            },
        )
        # Safely extract text even if no Parts are returned (e.g., blocked/empty candidates)
        text = ""
        try:
            if getattr(resp, "text", None):
                text = (resp.text or "").strip()
            else:
                # Fall back to concatenating any available candidate parts' text
                parts: List[str] = []
                for cand in getattr(resp, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    for p in getattr(content, "parts", []) or []:
                        part_text = getattr(p, "text", None)
                        if part_text:
                            parts.append(part_text)
                if parts:
                    text = "\n".join(parts).strip()
        except Exception:
            # If anything goes wrong during extraction, leave text as empty string
            text = ""
        self._cache.set(key, {"text": text})
        return text

    @retry_with_backoff(Exception)
    def choose(self, req: GenerationRequest) -> Dict[str, float]:
        if not req.choices or len(req.choices) == 0:
            raise ValueError("choices must be provided for choose()")

        key = self._cache_key("choose", req)
        cached = self._cache.get(key)
        if cached:
            return cached.get("probs", {})

        labeled: List[str] = []
        labels = []
        for i, choice in enumerate(req.choices):
            label = chr(ord("A") + i)
            labels.append(label)
            labeled.append(f"{label}) {choice}")

        instruction = (
            "Pick the single best option. Reply with the letter only (A, B, C, ...)."
        )
        prompt = f"{req.prompt}\n\nOptions:\n" + "\n".join(labeled) + f"\n\n{instruction}"

        select = self.generate(
            GenerationRequest(
                prompt=prompt,
                temperature=min(0.1, req.temperature),
                max_tokens=100,
                system_prompt=req.system_prompt,
                stop=req.stop,
            )
        )
        selected_label = select.strip().upper()[:1]
        probs = {c: 0.0 for c in req.choices}
        if selected_label in labels:
            idx = ord(selected_label) - ord("A")
            if 0 <= idx < len(req.choices):
                probs[req.choices[idx]] = 1.0

        self._cache.set(key, {"probs": probs})
        return probs

