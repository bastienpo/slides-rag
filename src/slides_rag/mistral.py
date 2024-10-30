"""MistralLM is a class that wraps the Mistral API."""

import dspy
from typing import Literal
import os
from mistralai import Mistral
from mistralai.utils import BackoffStrategy, RetryConfig


class MistralLM(dspy.LM):
    def __init__(
        self: "MistralLM",
        model: Literal["pixtral-12b-2409", "ministral-3b-2410"],
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """
        Initialize the MistralLM class.

        Args:
            model: The model to use.
            api_key: The mistral api key.
            temperature: The temperature to use.
            max_tokens: The max tokens to use.
        """
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")

        if api_key is None:
            raise ValueError("The mistral api key is not set and not provided.")

        super().__init__(model, **kwargs)

        self.history = []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Mistral(
            api_key=api_key,
            retry_config=RetryConfig(
                "backoff",
                BackoffStrategy(1, 50, 1.1, 100),
                False,
            ),
        )
        self.model = model

    def __call__(
        self: "MistralLM",
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Call the MistralLM class.

        Args:
            prompt: The prompt to use.
            messages: The messages to use.
            **kwargs: Additional keyword arguments to use.

        Returns:
            A list of strings.
        """
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        # Override the temperature and max tokens
        kwargs["temperature"] = self.temperature
        kwargs["max_tokens"] = self.max_tokens

        completions = self.client.chat.complete(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        self.history.append({"prompt": prompt, "completions": completions})

        return [completions.choices[0].message.content]

    def inspect_history(self: "MistralLM"):
        """
        Inspect the history of the MistralLM class.
        """
        for interaction in self.history:
            print(
                f"Prompt: {interaction['prompt']} -> Completions: {interaction['completions']}"
            )
