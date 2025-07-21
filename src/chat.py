import os

from langchain_openai import ChatOpenAI  # pip install langchain-openai


def get_llm():
    # """Return an OpenAI-compatible chat model served by openrouter.ai."""
    # api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is loaded from .env

    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY is not set in the .env file.")  # Check if API key exists

    return ChatOpenAI(
        model      = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key    = "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",
        base_url   = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        temperature= 0.2,
    )
