import json
import logging
import os
from typing import Any

import requests
import litellm

from agents.agent_output import AgentOutputSchemaBase
from agents.exceptions import ModelBehaviorError
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem
from agents.model_settings import ModelSettings
from agents.models.chatcmpl_converter import Converter
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from agents.tracing import generation_span
from agents.usage import Usage
from dotenv import load_dotenv
from openai import NotGiven
from openai.types.chat import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_function_tool_call import Function

load_dotenv()

logger = logging.getLogger(__name__)


BEARER_TOKEN = {
    "vnptai-hackathon-small": os.getenv("SMALL_BEARER_TOKEN"),
    "vnptai-hackathon-large": os.getenv("LARGE_BEARER_TOKEN"),
    "Qwen3-32B": ""
}

TOKEN_ID = {
    "vnptai-hackathon-small": os.getenv("SMALL_TOKEN_ID"),
    "vnptai-hackathon-large": os.getenv("LARGE_TOKEN_ID"),
    "Qwen3-32B": ""
    
}

TOKEN_KEY = {
    "vnptai-hackathon-small": os.getenv("SMALL_TOKEN_KEY"),
    "vnptai-hackathon-large": os.getenv("LARGE_TOKEN_KEY"),
    "Qwen3-32B": ""
    
}

BASE_URL = os.getenv("BASE_URL")


class InhouseModel(Model):
    """This class enables using any model via LiteLLM. LiteLLM allows you to acess OpenAPI,
    Anthropic, Gemini, Mistral, and many other models.
    See supported models here: [litellm models](https://docs.litellm.ai/docs/providers).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:

            _ = span_generation  # to avoid unused variable warning
            model = self.model

            headers = {
                "Authorization": f"Bearer {BEARER_TOKEN[model]}",
                "Token-id": TOKEN_ID[model],
                "Token-key": TOKEN_KEY[model],
                "Content-Type": "application/json"
            }

            converted_messages = Converter.items_to_messages(input)

            if system_instructions:
                converted_messages.insert(
                    0,
                    {
                        "content": system_instructions,
                        "role": "system",
                    },
                )

            response_format = Converter.convert_response_format(output_schema)
            tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)

            converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []
            for handoff in handoffs:
                converted_tools.append(Converter.convert_handoff_tool(handoff))

            json_data = {
                "model": model.replace("-", "_") if "Qwen" not in model else model,
                "messages": converted_messages,
                "temperature": 0,
                "top_p": model_settings.top_p,
                "top_k": None,
                "n": None,
                "stop": None,
                "max_completion_tokens": model_settings.max_tokens,
                "presence_penalty": model_settings.presence_penalty,
                "frequency_penalty": model_settings.frequency_penalty,
                "response_format": self._remove_not_given(response_format),
                "seed": None,
                "tools": converted_tools,
                "tool_choice": self._remove_not_given(tool_choice),
                "logprobs": None,
                "top_logprobs": None,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            json_data = {k: v for k, v in json_data.items() if v is not None}

            response = requests.post(
                f"{BASE_URL}/v1/chat/completions/{model}" if "Qwen" not in model else f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=json_data
            )

            response_data = json.loads(response.content)
            # response = {
            #     "prompt_logprobs": None,
            #     "created": 1765963491,
            #     "usage": {
            #         "completion_tokens": 73,
            #         "prompt_tokens": 102,
            #         "prompt_tokens_details": None,
            #         "total_tokens": 175
            #     },
            #     "kv_transfer_params": None,
            #     "model": "vnptai_hackathon_small",
            #     "service_tier": None,
            #     "id": "chatcmpl-772a0cc2f4084ba4ab5e5ce2fbc1551b",
            #     "choices": [
            #         {
            #             "token_ids": None,
            #             "finish_reason": "stop",
            #             "index": 0,
            #             "stop_reason": None,
            #             "message": {
            #                 "role": "assistant",
            #                 "function_call": None,
            #                 "refusal": None,
            #                 "annotations": None,
            #                 "tool_calls": [],
            #                 "audio": None,
            #                 "content": "Hello! I'm doing wonderfully, thank you for asking! 😊  \nI'm VNPTAI.IO, your friendly AI assistant developed by VNPT AI. I'm here to help you with any questions or tasks you might have—whether it's finding information, solving problems, or just having a chat. How can I assist you today? 🌟",
            #                 "reasoning_content": None
            #             },
            #             "logprobs": None
            #         }
            #     ],
            #     "system_fingerprint": None,
            #     "object": "chat.completion",
            #     "prompt_token_ids": None
            # }
            # logger.info(
            #     f"""LLM resp:\n{
            #         json.dumps(
            #             response.choices[0].message.model_dump(), indent=2, ensure_ascii=False
            #         )
            #     }\n"""
            # )
            items = Converter.message_to_output_items(
                LitellmConverter.convert_message_to_openai(response_data["choices"][0]["message"])
            )

            return ModelResponse(
                output=items,
                usage=Usage(),
                response_id=None,
            )

    def _remove_not_given(self, value: Any) -> Any:
        if isinstance(value, NotGiven):
            return None
        return value

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ):
        pass


class InternalChatCompletionMessage(ChatCompletionMessage):
    """
    An internal subclass to carry reasoning_content without modifying the original model.
    """

    reasoning_content: str


class LitellmConverter:
    @classmethod
    def convert_message_to_openai(
        cls, message
    ) -> ChatCompletionMessage:
        if message["role"] != "assistant":
            raise ModelBehaviorError(f"Unsupported role: {message['role']}")

        tool_calls: list[
            ChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
        ] | None = (
            [LitellmConverter.convert_tool_call_to_openai(tool) for tool in message["tool_calls"]]
            if message["tool_calls"]
            else None
        )

        # provider_specific_fields = message.get("provider_specific_fields", None)
        # refusal = (
        #     provider_specific_fields.get("refusal", None) if provider_specific_fields else None
        # )

        reasoning_content = ""
        # if hasattr(message, "reasoning_content") and message.reasoning_content:
        #     reasoning_content = message.reasoning_content

        return InternalChatCompletionMessage(
            content=message["content"],
            refusal=None,
            role="assistant",
            annotations=None,
            audio=message.get("audio", None),  # litellm deletes audio if not present
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    @classmethod
    def convert_tool_call_to_openai(
        cls, tool_call: litellm.types.utils.ChatCompletionMessageToolCall
    ) -> ChatCompletionMessageFunctionToolCall:
        return ChatCompletionMessageFunctionToolCall(
            id=tool_call["id"],
            type="function",
            function=Function(
                name=tool_call["function"]["name"],
                arguments=tool_call["function"]["arguments"],
            ),
        )  