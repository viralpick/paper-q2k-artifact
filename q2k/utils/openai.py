from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from q2k.config.openai_config import get_openai_configuration


_openai_config = get_openai_configuration()

_embeddings_model = OpenAIEmbeddings(
    api_key=_openai_config.OPENAI_API_KEY,
    model="text-embedding-3-large",
    dimensions=1536,
)


async def create_embeddings(text: str, dimensions: int = 1536) -> list[float]:
    """
    Create embeddings for the given text using a pre-configured model.
    """
    text = text.replace("\n", " ")

    return await _embeddings_model.aembed_query(text, dimensions=dimensions)


def get_openai_model(model_name: str):
    """
    Get the configured Bedrock model instance.
    """
    return ChatOpenAI(
        api_key=_openai_config.OPENAI_API_KEY,
        model=model_name,
        max_tokens=None,
    )
