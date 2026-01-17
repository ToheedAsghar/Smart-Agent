import time
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

# Set dummy env vars to allow imports
os.environ["OPENROUTER_API_KEY"] = "dummy"
os.environ["GEMINI_API_KEY"] = "dummy"
os.environ["TAVILY_API_KEY"] = "dummy"
os.environ["OPENAI_API_KEY"] = "dummy"

from src.agent.state import PlanState, PlanStep, AgentState
import src.agent.nodes as nodes

class TestConcurrency(unittest.TestCase):
    def test_parallel_execution(self):
        print("Starting test_parallel_execution")
        # Setup Plan with 2 parallel steps
        step1 = PlanStep(step_id=1, description="Task 1", tool_required=True, section_id=1)
        step2 = PlanStep(step_id=2, description="Task 2", tool_required=True, section_id=1)
        step3 = PlanStep(step_id=3, description="Task 3", tool_required=False, section_id=2)

        plan = PlanState(steps=[step1, step2, step3], rationale="Test plan")

        state = AgentState(
            messages=[],
            plan=plan,
            current_step=0,
            step_results={},
            final_output=None,
            reflection=None,
            retry_cnt=0
        )

        # Mock tool to sleep
        mock_tool = MagicMock()
        def slow_tool(arg):
            time.sleep(1.0) # Sleep 1 second
            return f"Result for {arg}"

        mock_tool.invoke.side_effect = slow_tool

        # Patch tool_map and tool_chain in nodes module
        # We need to mock tool_chain.invoke to return a message that triggers the tool
        mock_chain = MagicMock()
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{'name': 'mock_tool', 'args': 'some_arg'}]
        mock_chain.invoke.return_value = mock_msg

        # Apply patches
        with patch.object(nodes, 'tool_map', {'mock_tool': mock_tool}), \
             patch.object(nodes, 'tool_chain', mock_chain):

            start_time = time.time()
            result_state = nodes.node_executor(state)
            end_time = time.time()

            duration = end_time - start_time
            print(f"Execution took {duration:.2f} seconds")

            # If parallel, it should take ~1.0s, not ~2.0s
            # Allow some overhead
            self.assertLess(duration, 1.8, "Execution took too long, likely serial")
            self.assertGreater(duration, 0.9, "Execution took too short")

            # Check results
            # idx 0 and 1 should be in results
            self.assertIn(0, result_state['step_results'])
            self.assertIn(1, result_state['step_results'])
            self.assertEqual(len(result_state['step_results']), 2)

            # Check current_step updated correctly
            self.assertEqual(result_state['current_step'], 2)

            # Check if both tools were called
            self.assertEqual(mock_tool.invoke.call_count, 2)

if __name__ == '__main__':
    unittest.main()
