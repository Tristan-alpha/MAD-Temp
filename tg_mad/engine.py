"""Custom VLLMEngine for TextGrad that routes to specific vLLM server endpoints.

TextGrad's built-in LiteLLMEngine does not pass api_base to litellm.completion(),
making it impossible to route to different vLLM servers on different ports.
This engine inherits from the same base class and adds api_base support.
"""

import os
from typing import Union, List

import diskcache as dc
from litellm import completion
from textgrad.engine_experimental.base import EngineLM, cached
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tg_mad.config import MAX_NEW_TOKENS, TOP_P


class VLLMEngine(EngineLM):
    """Engine that routes LLM calls to a specific vLLM OpenAI-compatible server."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str,
        base_url: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 1.0,
        max_tokens: int = MAX_NEW_TOKENS,
        top_p: float = TOP_P,
        is_multimodal: bool = False,
        cache: Union[dc.Cache, bool] = False,
    ):
        """
        :param model_string: litellm model string, e.g. "hosted_vllm/Qwen/Qwen3-4B-Instruct-2507"
        :param base_url: vLLM server URL, e.g. "http://localhost:8000/v1"
        :param temperature: Sampling temperature (1.0 for debater, 0.7 for evaluator)
        :param max_tokens: Max tokens for generation
        """
        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def _generate_from_single_prompt(
        self,
        content: str,
        system_prompt: str = None,
        temperature=None,
        max_tokens=None,
        top_p=None,
    ):
        sys_prompt = system_prompt or self.system_prompt
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        nucleus = top_p if top_p is not None else self.top_p

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ]

        response = completion(
            model=self.model_string,
            messages=messages,
            api_base=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            temperature=temp,
            max_tokens=tokens,
            top_p=nucleus,
        )
        return response["choices"][0]["message"]["content"]

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
    ):
        raise NotImplementedError(
            "VLLMEngine does not support multimodal input. Use text-only prompts."
        )

    def __call__(self, content, **kwargs):
        # Base class __call__ is a no-op; must override to delegate to generate()
        return self.generate(content, **kwargs)


def create_debater_engine(
    model: str = None,
    base_url: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> VLLMEngine:
    """Factory for the debater engine (Qwen3-4B)."""
    from tg_mad.config import DEBATER_MODEL, DEBATER_BASE_URL, TEMPERATURE

    return VLLMEngine(
        model_string=model or DEBATER_MODEL,
        base_url=base_url or DEBATER_BASE_URL,
        temperature=temperature if temperature is not None else TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else MAX_NEW_TOKENS,
        cache=False,  # Need stochastic responses for agent diversity
    )


def create_evaluator_engine(
    model: str = None,
    base_url: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> VLLMEngine:
    """Factory for the evaluator/backward engine (Qwen3-8B)."""
    from tg_mad.config import EVALUATOR_MODEL, EVALUATOR_BASE_URL, EVALUATOR_TEMPERATURE

    return VLLMEngine(
        model_string=model or EVALUATOR_MODEL,
        base_url=base_url or EVALUATOR_BASE_URL,
        temperature=temperature if temperature is not None else EVALUATOR_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else MAX_NEW_TOKENS,
        cache=False,
    )
