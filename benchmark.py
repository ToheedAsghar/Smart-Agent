import sys
import time
import asyncio
from unittest.mock import MagicMock

# 1. Setup Mocks BEFORE importing src modules
# Mock the config module
mock_config = MagicMock()
sys.modules["src.utils.config"] = mock_config

# Mock the tools module
mock_tools_mod = MagicMock()
sys.modules["src.tools.custom_tools"] = mock_tools_mod

# Mock LLM and Tools
class MockLLM:
    def __init__(self, name="MockLLM"):
        self.name = name

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, input, *args, **kwargs):
        # Simulate blocking I/O
        time.sleep(0.5)
        return self._generate_response(input)

    async def ainvoke(self, input, *args, **kwargs):
        # Simulate non-blocking I/O
        await asyncio.sleep(0.5)
        return self._generate_response(input)

    def _generate_response(self, input):
        # Return a structure that satisfies the node logic
        # For planner (returns PlanState)
        # For executor (returns string or tool output)
        # For reflector (returns ReflectorState)
        # For synthesizer (returns string)

        # We can try to infer type from input or context,
        # but simpler to return a MagicMock that can be anything.
        resp = MagicMock()
        resp.content = "Mock Content"
        resp.tool_calls = [] # Default no tool calls

        # For Planner node, it expects an object with .steps
        resp.steps = [
            MagicMock(description="Step 1", tool_required=False, step_id=1),
            MagicMock(description="Step 2", tool_required=False, step_id=2)
        ]
        resp.rationale = "Mock Rationale"

        # For Reflector node, it expects .is_satisfactory and .feedback
        resp.is_satisfactory = True
        resp.feedback = "Good job"

        return resp

mock_llm_instance = MockLLM()
mock_config.get_llm.return_value = mock_llm_instance
mock_config.get_reasoner_llm.return_value = mock_llm_instance
mock_tools_mod.get_tools.return_value = []

# 2. Import the app (this will trigger node imports which use the mocks)
try:
    from src.agent.graph import app
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

# 3. Benchmark Functions

def run_sync_benchmark():
    print("Running Sync Benchmark (Sequential)...")
    # We simulate 3 sequential requests.
    # In a real sync server, requests are processed one by one (or in threads).
    # Since we want to show 'blocking' behavior, sequential is the worst case baseline.
    # Even with threads, if the underlying call holds a lock or if we are just single-process,
    # it blocks. Here we show single-thread sequential time.

    start_time = time.time()
    for i in range(3):
        print(f"  Request {i+1} starting...")
        try:
            # We use a dummy message
            app.invoke({"messages": [HumanMessage(content="Test query")]})
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Sync Benchmark Complete. Total Time: {total_time:.4f}s")

async def run_async_benchmark():
    print("Running Async Benchmark (Concurrent)...")

    start_time = time.time()
    tasks = []
    for i in range(3):
        tasks.append(run_single_async_request(i))

    await asyncio.gather(*tasks)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Async Benchmark Complete. Total Time: {total_time:.4f}s")

async def run_single_async_request(i):
    print(f"  Request {i+1} starting...")
    try:
        # app.ainvoke is available on CompiledGraph
        await app.ainvoke({"messages": [HumanMessage(content="Test query")]})
    except Exception as e:
        print(f"  Request {i+1} failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "async":
        asyncio.run(run_async_benchmark())
    else:
        run_sync_benchmark()
