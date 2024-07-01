import asyncio
import functools
import json
import os
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Literal, Optional

import openai
from unify.chat import ChatBot as ChatBotClient
from unify.clients import AsyncUnify as AsyncUnifyClient
from unify.clients import Unify as UnifyClient
from unify.exceptions import status_error_map

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM

try:
    import openai.error
    from openai.openai_object import OpenAIObject

    ERRORS = (openai.error.RateLimitError,)
except Exception:
    ERRORS = (openai.RateLimitError,)
    OpenAIObject = dict


class ToAsync:
    def __init__(self, *, run_async: Optional[bool] = None, executor: Optional[ThreadPoolExecutor] = None):
        self.executor = executor
        self.run_async = run_async

    def __call__(self, blocking, **kwargs):
        run_async = kwargs.get("run_async")
        assert run_async is bool or run_async is None, "run_async parameter expected type is [bool]"
        self.run_async = run_async

        @wraps(blocking)
        async def wrapper(*args: Any, **kwargs: Any):
            loop = asyncio.get_event_loop()
            if not self.executor:
                self.executor = ThreadPoolExecutor()

            func = partial(blocking, *args, **kwargs)

            return await loop.run_in_executor(self.executor, func)

        if self.run_async:
            return wrapper
        return blocking


class AsyncUnifyClient(AsyncUnifyClient):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(endpoint, model, provider, api_key)

    async def async_generate_completion(self, endpoint, messages, max_tokens, stream) -> Any:
        return await self.client.chat.completions.create(
            model=endpoint,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            stream=stream,
        )

    async def _generate_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            async_stream = self.async_generate_completion(
                endpoint,
                messages,  # type: ignore[arg-type]
                max_tokens,
                True,
            )
            async for chunk in async_stream:  # type: ignore[union-attr]
                self.set_provider(chunk.model.split("@")[-1])
                yield chunk.choices[0].delta.content or ""
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None

    async def _generate_non_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            async_response = self.async_generate_completion(
                endpoint,
                messages,  # type: ignore[arg-type]
                max_tokens,
                True,
            )
            self.set_provider(async_response.model.split("@")[-1])  # type: ignore
            return async_response.choices[0].message.content.strip(" ")  # type: ignore # noqa: E501, WPS219
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None


class ChatBotClient(ChatBotClient):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(endpoint, model, provider, api_key)

    def generate_completion(
        self,
        inp: str,
    ) -> Generator[str, None, None]:
        """Processes the user input to generate AI response."""
        self._update_message_history(role="user", content=inp)
        return self._client.generate(
            messages=self._message_history,
            stream=True,
        )


class UnifyClient(UnifyClient):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(endpoint, model, provider, api_key)

    def generate_completion(
        self,
        endpoint: str = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = 1024,
        stream: Optional[bool] = True,
        **kwargs,
    ) -> Any:
        return self.client.chat.completions.create(
            model=endpoint,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            stream=stream,
            name="Unify-generation",
            metadata={
                "model": self.model,
                "provider": self.provider,  # todo: update trace metadata after call (see set_provider)
                "endpoint": self.endpoint,
            },
            **kwargs,
        )

    def _generate_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            chat_completion = completions_request(
                self,
                endpoint=endpoint,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in chat_completion:
                content = chunk.choices[0].delta.content  # type: ignore[union-attr]
                self.set_provider(chunk.model.split("@")[-1])  # type: ignore[union-attr]
                if content is not None:
                    yield content
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None

    def _generate_non_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            chat_completion = completions_request(
                self,
                endpoint=endpoint,
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )

            self.set_provider(
                chat_completion.model.split(  # type: ignore[union-attr]
                    "@",
                )[-1],
            )

            return chat_completion.choices[0].message.content.strip(" ")  # type: ignore # noqa: E501, WPS219
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None


class Unify(LM):
    def __init__(
        self,
        endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        api_key=None,
        **kwargs,  # Added to accept additional keyword arguments
    ):
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.api_provider: Literal["unify"] = "unify"
        self.api_base = "https://api.unify.ai/v0"
        self.model = model
        self.provider = provider
        super().__init__(model=self.endpoint)
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1,
            "top_k": 20,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "num_ctx": 1024,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def __call__(
        self,
        prompt: Optional[str] = "",
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from the model called by unify.

        Args:
            prompt (str): prompt to send to unify
            only_completed (bool, optional): return only completed responses
                and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using
                the returned probabilities. Defaults to False.
            **kwargs (Any): metadata passed to the model.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        assert prompt, "for now"
        assert kwargs, "for now"
        unify_client = UnifyClient(
            endpoint=self.endpoint,
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
        )
        return unify_client.generate(user_prompt=prompt, system_prompt=self.system_prompt, **self.kwargs)

    @ToAsync()
    def basic_request(self, prompt: str, **kwargs) -> Any:
        """Send request to the Unify AI API. This method is required by the LM base class."""
        run_async = kwargs.get("run_async")
        kwargs = {**self.kwargs, **kwargs}

        settings_dict = {
            "model": self.model,
            "options": {k: v for k, v in kwargs.items() if k not in ["n", "max_tokens"]},
            "stream": False,
        }
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            settings_dict["prompt"] = prompt

        # Call the generate method
        return self._call_generate(settings_dict) if not run_async else self._call_generate_async(settings_dict)

    async def _call_generate_async(self, settings_dict):
        """Call the generate method from the AsyncUnify client."""
        unify_instance = AsyncUnifyClient()

        try:
            return unify_instance.async_generate_completion(settings=settings_dict, api_key=self.api_key)
        except Exception as e:
            return f"An error occurred while calling the generate method: {e}"

    def _call_generate(self, settings_dict):
        """Call the generate method from the unify client."""
        try:
            return completions_request(settings=settings_dict, api_key=self.api_key)
        except Exception as e:
            return f"An error occurred while calling the generate method: {e}"


@CacheMemory.cache
def unify_cached(unify_instance: Optional[Unify], **kwargs) -> OpenAIObject:
    endpoint = kwargs.get("endpoint")
    model = kwargs.get("model")
    api_key = kwargs.get("api_key")
    provider = kwargs.get("provider")
    unify_instance = (
        unify_instance
        if unify_instance
        else UnifyClient(endpoint=endpoint, model=model, provider=provider, api_key=api_key)
    )
    return unify_instance.generate_completion(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def unify_cached_wrapped(unify_instance: Optional[Unify], **kwargs) -> OpenAIObject:
    return unify_cached(unify_instance, **kwargs)


def completions_request(unify_instance: Optional[Unify], **kwargs) -> dict:
    return unify_cached_wrapped(unify_instance, **kwargs).model_dump()


@CacheMemory.cache
def unify_chatbot_cached_request(**kwargs) -> OpenAIObject:
    endpoint = kwargs.get("endpoint")
    model = kwargs.get("model")
    provider = kwargs.get("provider")
    api_key = kwargs.get("api_key")
    unify_chatbot_client = ChatBotClient(endpoint=endpoint, model=model, provider=provider, api_key=api_key)
    inp = kwargs.get("messages")
    if inp:
        del kwargs["messages"]
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return unify_chatbot_client.generate_completion(inp, **kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def unify_chatbot_cached_request_wrapped(**kwargs) -> OpenAIObject:
    return unify_chatbot_cached_request(**kwargs)


def chat_request(**kwargs) -> dict:
    return unify_chatbot_cached_request_wrapped(**kwargs).model_dump()


# Usage example
if __name__ == "__main__":
    # Initialize the UnifyAI instance with a specific model and fallback
    unify_lm = Unify(endpoint="llama-3-8b-chat@fireworks-ai->gpt-3.5-turbo@openai")

    # Check credit balance
    credit_balance = unify_lm.get_credit_balance()
    print(f"Current credit balance: {credit_balance}")  # type: ignore # noqa: T201

    # List available models
    print("Available models:")  # type: ignore # noqa: T201
    models = unify_lm.list_available_models()
    for model in models:
        print(f"- {model}")  # type: ignore # noqa: T201

    # Generate a response
    prompt = "Translate 'Hello, world!' to French."
    print(f"\nGenerating response for prompt: '{prompt}'")  # type: ignore # noqa: T201
    responses = unify_lm.generate(prompt, max_tokens=50, temperature=0.7, n=1)

    if responses:
        print("Generated response:")  # type: ignore # noqa: T201
        for response in responses:
            print(response)  # type: ignore # noqa: T201
    else:
        print("Failed to generate any responses.")  # type: ignore # noqa: T201

    # Example with router
    router_lm = Unify(endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03")
    print("\nUsing router for generation:")  # type: ignore # noqa: T201
    router_responses = router_lm.generate("What is the capital of France?", max_tokens=50)
    if router_responses:
        print("Router-generated response:")  # type: ignore # noqa: T201
        for response in router_responses:
            print(response)  # type: ignore # noqa: T201
    else:
        print("Router failed to generate any responses.")  # type: ignore # noqa: T201
