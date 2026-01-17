import sys
import os
import unittest
from unittest.mock import MagicMock

# Set dummy env vars
os.environ["OPENROUTER_API_KEY"] = "dummy"
os.environ["GEMINI_API_KEY"] = "dummy"
os.environ["TAVILY_API_KEY"] = "dummy"

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules before import
mock_config = MagicMock()
mock_tools_module = MagicMock()

sys.modules['src.utils.config'] = mock_config
sys.modules['src.tools.custom_tools'] = mock_tools_module

# Setup mocks for get_llm, etc.
mock_llm = MagicMock()
mock_config.get_llm.return_value = mock_llm
mock_config.get_reasoner_llm.return_value = MagicMock()
mock_tools_module.get_tools.return_value = []

# Now import the module under test
from src.agent import nodes
from src.agent.state import PlanStep, PlanState

class TestExecutor(unittest.TestCase):

    def setUp(self):
        self.mock_tool1 = MagicMock()
        self.mock_tool1.invoke.return_value = "Result 1"
        self.mock_tool1.name = "tool1"

        self.mock_tool2 = MagicMock()
        self.mock_tool2.invoke.return_value = "Result 2"
        self.mock_tool2.name = "tool2"

        # Patch the global tool_map in nodes module
        nodes.tool_map = {
            "tool1": self.mock_tool1,
            "tool2": self.mock_tool2
        }

        # Patch the global tool_chain in nodes module
        self.mock_tool_chain = MagicMock()
        nodes.tool_chain = self.mock_tool_chain

    def test_multiple_tool_calls(self):
        # Setup state
        step = PlanStep(step_id=0, description="Task", tool_required=True)
        plan = PlanState(steps=[step], rationale="test")
        state = {
            'plan': plan,
            'current_step': 0,
            'step_results': {},
            'retry_cnt': 0,
            'messages': []
        }

        # Setup mock tool chain response with 2 calls
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {'name': 'tool1', 'args': {'arg': '1'}},
            {'name': 'tool2', 'args': {'arg': '2'}}
        ]
        self.mock_tool_chain.invoke.return_value = mock_response

        # Run executor
        print("Running node_executor...")
        result_state = nodes.node_executor(state)

        # Check results
        step_results = result_state['step_results'][0]
        print(f"Step Results: {step_results}")

        # Verification
        # 1. Verify tool1 called
        self.mock_tool1.invoke.assert_called()

        # 2. Verify tool2 called
        self.mock_tool2.invoke.assert_called()

        # 3. Verify output contains both
        self.assertIn("Result 1", step_results)
        self.assertIn("Result 2", step_results)

if __name__ == '__main__':
    unittest.main()
