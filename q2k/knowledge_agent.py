import logging
from typing import Any, Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from q2k.utils.llm_helper import invoke_async_with_backoff, parse_ai_message
from q2k.utils.openai import get_openai_model
from q2k.utils.parser import strip_url_annotations


logger = logging.getLogger(__name__)

_openai_model_name = "gpt-5-mini"
_openai_model = get_openai_model(_openai_model_name)
tool = {"type": "web_search_preview"}
_knowledge_model = _openai_model.bind_tools([tool], tool_choice="required")


class _KnowledgeItem(BaseModel):
    """
    DTO representing a single knowledge item search result.
    """

    id: str
    question: str
    knowledge: str
    llm_agent: str
    task: str
    input_text: str
    embedding: list[float]
    additional_info: Optional[dict[str, Any]]
    is_inspected: bool = False


async def run_knowledge_agent(
    model_input: str, question: str, task_description: str
) -> tuple[_KnowledgeItem, int, int]:
    """
    Run the knowledge agent to retrieve information based on the input and question.
    Returns a tuple containing the knowledge item, input token count, and output token count.
    """

    _system_prompt_text_path = "prompts/knowledge.txt"

    with open(_system_prompt_text_path, "r", encoding="utf-8") as f:
        _system_prompt = f.read()
        f.close()

    llm_agent = "Product Matching Agent"

    _human_prompt = f"""Task: {task_description}
Input: {model_input}
Question: {question}"""
    messages = [
        SystemMessage(content=_system_prompt),
        HumanMessage(content=_human_prompt),
    ]

    response = await invoke_async_with_backoff(_knowledge_model.ainvoke, messages)
    parsed_response = parse_ai_message(response)

    text = parsed_response.get("text", "")
    parsed_text = strip_url_annotations(text)

    knowledge_item = _KnowledgeItem(
        id=str(uuid4()),
        question=question,
        knowledge=parsed_text,
        llm_agent=llm_agent,
        task=task_description,
        input_text=model_input,
        embedding=[],
        is_inspected=False,
        additional_info={
            "model_name": _openai_model_name,
            "annotations": parsed_response.get("annotations", []),
        },
    )

    input_tokens = response.usage_metadata.get("input_tokens", 0)
    output_tokens = response.usage_metadata.get("output_tokens", 0)

    return knowledge_item, input_tokens, output_tokens
