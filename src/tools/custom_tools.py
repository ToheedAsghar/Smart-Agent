from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from simpleeval import SimpleEval

# Initialize the evaluator at module level for performance (avoids re-instantiation).
# We limit the allowed functions and disable all variable access for security.
math_evaluator = SimpleEval()
math_evaluator.functions = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
math_evaluator.names = {} # No variables allowed

@tool
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression. Input should be a valid python math expression string."""
    try:
        return str(math_evaluator.eval(expression))
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def get_tools():
    """Returns a list of tools available for the agent."""

    # 1. Tavily Search
    # This requires TAVILY_API_KEY in env
    search_tool = TavilySearch(max_results=2)

    # 2. Wikipedia
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # 3. Calculator
    # calculator is defined at module level

    return [search_tool, wikipedia, calculator]
