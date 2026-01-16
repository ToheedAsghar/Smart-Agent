from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

def get_tools():
    """Returns a list of tools available for the agent."""

    # 1. Tavily Search
    # This requires TAVILY_API_KEY in env
    search_tool = TavilySearch(max_results=2)

    # 2. Wikipedia
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # 3. Calculator
    @tool
    def calculator(expression: str) -> str:
        """Calculate the result of a mathematical expression. Input should be a valid python math expression string."""
        try:
            # Note: eval is used here for simplicity. In a real constrained env, use a safer parser.
            # Limiting globals/locals for safety
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
            return str(eval(expression, {"__builtins__": None}, allowed_names))
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"

    return [search_tool, wikipedia, calculator]
