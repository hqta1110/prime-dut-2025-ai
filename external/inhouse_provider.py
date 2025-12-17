"""
Custom LiteLLM Provider that doesn't require OpenAI environment variables.
This provider directly passes the LiteLLM credentials to the model.
"""
import os
from inhouse_model import InhouseModel
# from agents.extensions.models.litellm_model import LitellmModel
from agents.models.default_models import get_default_model
from agents.models.interface import Model, ModelProvider


class InhouseProvider(ModelProvider):
    """
    A custom LiteLLM provider that uses LiteLLM credentials directly.

    Usage:
    ```python
    Runner.run(agent, input, run_config=RunConfig(model_provider=CustomLitellmProvider()))
    ```
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the provider with LiteLLM credentials.

        Args:
            api_key: LiteLLM API key. If None, will use LITELLM_API_KEY environment variable.
            base_url: LiteLLM base URL. If None, will use LITELLM_BASE_URL environment variable.
        """
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL")

        if not self.api_key:
            raise ValueError("LiteLLM API key is required. Set LITELLM_API_KEY environment variable or pass api_key parameter.")

        if not self.base_url:
            raise ValueError("LiteLLM base URL is required. Set LITELLM_BASE_URL environment variable or pass base_url parameter.")

    def get_model(self, model_name: str | None) -> Model:
        """
        Get a LiteLLM model with the configured credentials.

        Args:
            model_name: Name of the model to use. If None, uses default model.

        Returns:
            LitellmModel instance configured with the credentials.
        """
        return InhouseModel(
            model=model_name or get_default_model(),
            api_key=self.api_key,
            base_url=self.base_url
        )
