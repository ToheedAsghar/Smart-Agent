import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    """Initializes and returns the main LLM for planning and execution."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    # If using OpenRouter, we need a base_url. If using standard OpenAI, base_url is None.
    # The original backend used OpenRouter.
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        api_key=api_key
    )

def get_reasoner_llm():
    """Initializes and returns the reasoning LLM for reflection."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
