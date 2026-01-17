import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Set dummy API keys before importing modules that initialize LLMs
os.environ["OPENROUTER_API_KEY"] = "dummy"
os.environ["GEMINI_API_KEY"] = "dummy"
os.environ["TAVILY_API_KEY"] = "dummy"

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.nodes import node_synthesizer
from src.agent.state import AgentState
from langchain_core.messages import AIMessage, HumanMessage

class TestNodes(unittest.TestCase):
    @patch('src.agent.nodes.llm')
    def test_node_synthesizer(self, mock_llm):
        # Mock LLM response
        mock_response = AIMessage(content="Final Answer")
        mock_llm.invoke.return_value = mock_response

        # Setup state
        state = {
            'messages': [HumanMessage(content="What is the weather?")],
            'step_results': {
                0: "It is sunny.",
                1: "Temperature is 25C."
            }
        }

        # Run function
        result = node_synthesizer(state)

        # Verify result
        self.assertEqual(result['final_output'], "Final Answer")

        # Verify LLM was called with correct context
        args, _ = mock_llm.invoke.call_args
        prompt = args[0]

        # Check if the generator expression worked and joined the strings correctly
        expected_context_part = "info found: It is sunny.\ninfo found: Temperature is 25C."
        self.assertIn(expected_context_part, prompt)

if __name__ == '__main__':
    unittest.main()
