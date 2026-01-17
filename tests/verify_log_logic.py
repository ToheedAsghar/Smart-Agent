import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import state classes
try:
    from src.agent.state import PlanState, ReflectorState, PlanStep
except ImportError as e:
    print(f"Error importing state classes: {e}")
    sys.exit(1)

def verify_log_logic():
    print("Starting verification of log logic...")

    # Mock data
    mock_plan = PlanState(
        rationale="Test rationale",
        steps=[
            PlanStep(step_id=1, description="Step 1", tool_required=True),
            PlanStep(step_id=2, description="Step 2", tool_required=False)
        ]
    )

    mock_reflection = ReflectorState(
        is_satisfactory=False,
        feedback="Test feedback",
        next_step_adjustment="Adjust"
    )

    # Simulate app.stream events
    events = [
        {"planner": {"plan": mock_plan}},
        {"executor": {"current_step": 1, "step_results": {0: "Result 1"}}},
        {"reflector": {"reflection": mock_reflection}},
        {"synthesizer": {"final_output": "Final Answer"}}
    ]

    # Initialize process history
    process_history = []
    final_answer = ""

    # Simulate the loop in app.py
    for event in events:
        # --- VISUALIZE PLANNING ---
        if "planner" in event:
            plan = event["planner"]["plan"]
            # Logic to be implemented in app.py
            process_history.append({
                "type": "planner",
                "rationale": plan.rationale,
                "steps": [s.model_dump() if hasattr(s, 'model_dump') else s.dict() for s in plan.steps]
            })

        # --- VISUALIZE EXECUTION ---
        if "executor" in event:
            idx = event["executor"]["current_step"] - 1
            result = event["executor"]["step_results"][idx]

            process_history.append({
                "type": "executor",
                "step": idx + 1,
                "result": result
            })

        # --- VISUALIZE REFLECTION ---
        if "reflector" in event:
            reflection = event["reflector"]["reflection"]
            process_history.append({
                "type": "reflector",
                "is_satisfactory": reflection.is_satisfactory,
                "feedback": reflection.feedback
            })

        # --- VISUALIZE FINAL OUTPUT ---
        if "synthesizer" in event:
            final_answer = event["synthesizer"]["final_output"]

    # Verify process_history structure
    print("\nCaptured Process History:")
    print(json.dumps(process_history, indent=2))

    assert len(process_history) == 3, "Should have 3 events"

    # Check planner
    assert process_history[0]["type"] == "planner"
    assert process_history[0]["rationale"] == "Test rationale"
    assert len(process_history[0]["steps"]) == 2
    assert process_history[0]["steps"][0]["step_id"] == 1

    # Check executor
    assert process_history[1]["type"] == "executor"
    assert process_history[1]["step"] == 1
    assert process_history[1]["result"] == "Result 1"

    # Check reflector
    assert process_history[2]["type"] == "reflector"
    assert process_history[2]["is_satisfactory"] is False
    assert process_history[2]["feedback"] == "Test feedback"

    print("\nVerification successful!")

if __name__ == "__main__":
    verify_log_logic()
