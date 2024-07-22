from typing import Any, Optional

from unify.clients import Unify as UnifyClient

from dsp.modules.lm import LM


class Unify(LM):
    """A class to interact with the Unify AI API."""

    def __init__(
        self,
        model: Optional[str] = None,
        stream: Optional[bool] = False,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.client = UnifyClient(api_key=self.api_key, endpoint=self.model)

        super().__init__(model=self.model)
        self.provider = "unify"
        self.stream = stream
        self.system_prompt = system_prompt
        self.kwargs = {
            "model": self.model,
            "temperature": 0.8,
            "max_tokens": 150,
            "top_p": 1,
            "n": 1,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        """Basic request to the Unify's API."""
        kwargs = {**self.kwargs, **kwargs}
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        raw_response = self.client.generate(
            messages=messages,
            stream=self.stream,
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
        )
        response_content = (
            raw_response if isinstance(raw_response, str) else "".join(raw_response)
        )  # handle stream=True or False outputs from Unify generate method
        formated_response: dict = {"choices": [{"message": {"content": response_content}}]}

        history = {
            "prompt": prompt,
            "response": formated_response,
            "kwargs": kwargs,
        }

        self.history.append(history)

        return formated_response

    def __call__(
        self,
        prompt: Optional[str],
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Request completions from the Unify API."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n: int = kwargs.pop("n", 1)
        completions = []

        for _ in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response["choices"][0]["message"]["content"])

        return completions
