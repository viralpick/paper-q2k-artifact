import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from q2k.utils.llm_helper import invoke_async_with_backoff
from q2k.utils.openai import get_openai_model

logger = logging.getLogger(__name__)


class _ReasonedResponse(BaseModel):
    """
    Segmented chain‐of‐thought output from the LLM,
    broken into discrete reasoning steps.
    """

    thinking: str
    questions: list[str]


_MODEL_NAME = "gpt-5-mini"
_model = get_openai_model(_MODEL_NAME)
_extractor_model = _model.with_structured_output(
    _ReasonedResponse,
    include_raw=True,
)


async def run_reasoning_agent(base_product: str, compared_product: str) -> dict:
    """
    Run the reasoning agent to compare two products and questions.
    Returns a dictionary containing the reasoning output and token usage.
    """
    _system_prompt_text_path = "prompts/reasoning.txt"

    with open(_system_prompt_text_path, "r", encoding="utf-8") as f:
        _system_prompt = f.read()
        f.close()

    _human_prompt = HumanMessage(
        content=(f"base product: {base_product} / compared product: {compared_product}")
    )

    messages = [_system_prompt, _human_prompt]
    llm_response = await invoke_async_with_backoff(_extractor_model.ainvoke, messages)

    raw_response = llm_response["raw"]
    parsed_response = llm_response["parsed"]

    input_tokens = raw_response.usage_metadata["input_tokens"]
    output_tokens = raw_response.usage_metadata["output_tokens"]

    return {
        "base_product": base_product,
        "compared_product": compared_product,
        "input": _human_prompt.content,
        "thinking": parsed_response.thinking,
        "questions": parsed_response.questions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_name": _MODEL_NAME,
    }
