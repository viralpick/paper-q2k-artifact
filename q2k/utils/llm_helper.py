import asyncio
import logging
from openai import APIStatusError
from typing import Any, Callable

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


async def invoke_async_with_backoff(
    fn: Callable, *args, max_retries: int = 10, base_delay: float = 10, **kwargs
):
    """
    Async version of invoke_with_backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except APIStatusError as e:
            if attempt == max_retries:
                raise
            delay = base_delay  # (2 ** (attempt - 1))
            logger.warning(
                f"ThrottlingException encountered. Retrying in {delay:.2f} seconds... (Attempt {attempt}/{max_retries})"
            )
            logger.error("Exception details: %s", e)
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed to invoke function after {max_retries} attempts.")


def parse_ai_message(msg: AIMessage) -> dict[str, Any]:
    """
    Parse an AIMessage object to extract relevant information.
    """
    # Try property first (newer LangChain versions), then method (deprecated), then fallback
    try:
        text = msg.text  # property (current standard)
    except (AttributeError, TypeError):
        try:
            text = msg.text()  # method (deprecated)
        except Exception:
            text = getattr(msg, "content", "")

    usage = msg.usage_metadata

    input_tokens = usage.get("input_tokens", -1)
    output_tokens = usage.get("output_tokens", -1)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    annotations = None
    raw_content = getattr(msg, "content", msg)
    if isinstance(raw_content, list):
        anns = []
        for chunk in raw_content:
            if isinstance(chunk, dict) and "annotations" in chunk:
                anns.extend(chunk.get("annotations", []))
        if anns:
            annotations = anns

    return {
        "text": text,
        "annotations": annotations,
        "usage_raw": usage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw_message": msg,
    }
