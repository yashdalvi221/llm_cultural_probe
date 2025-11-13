import os
from typing import Dict, List

from .base import GenerationRequest, ModelClient
from ..utils.cache import ContentHashCache
from ..utils.rate_limit import retry_with_backoff


class OpenRouterClient(ModelClient):
	def __init__(
		self,
		model_id: str = "openrouter/auto",
		name: str = "openrouter-auto",
		cache_dir: str = ".cache",
	) -> None:
		super().__init__(name=name, model_id=model_id)
		self._cache = ContentHashCache(cache_dir=cache_dir)
		api_key = os.getenv("OPENROUTER_API_KEY")
		if not api_key:
			raise RuntimeError("OPENROUTER_API_KEY not set")
		# Lazy import to avoid hard dependency during docs builds
		import openai  # type: ignore

		self._client = openai.OpenAI(
			api_key=api_key,
			base_url="https://openrouter.ai/api/v1",
		)

	def _cache_key(self, kind: str, req: GenerationRequest) -> str:
		payload = {
			"provider": "openrouter",
			"kind": kind,
			"model_id": self.model_id,
			"prompt": req.prompt,
			"choices": req.choices,
			"temperature": req.temperature,
			"max_tokens": req.max_tokens,
			"seed": req.seed,
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

		messages = []
		if req.system_prompt:
			messages.append({"role": "system", "content": req.system_prompt})
		messages.append({"role": "user", "content": req.prompt})

		resp = self._client.chat.completions.create(
			model=self.model_id,
			messages=messages,
			temperature=req.temperature,
			max_tokens=req.max_tokens,
			seed=req.seed,
			stop=req.stop,
		)
		text = (resp.choices[0].message.content or "").strip()
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

		# Force deterministic selection via low temperature
		select = self.generate(
			GenerationRequest(
				prompt=prompt,
				temperature=min(0.1, req.temperature),
				max_tokens=20,
				seed=req.seed,
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


