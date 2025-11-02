from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class OpenAIConfiguration(BaseSettings):

    OPENAI_API_KEY: str

    model_config = ConfigDict(env_file=".env", extra="ignore")


def get_openai_configuration() -> OpenAIConfiguration:
    _openai_configuration = OpenAIConfiguration()
    """
    Get the OpenAI configuration settings.
    """
    return _openai_configuration
