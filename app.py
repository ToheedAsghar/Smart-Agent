import streamlit as st
import json
from langchain_core.messages import HumanMessage
from src.agent.graph import app
from src.agent.state import PlanState, ReflectorState

st.set_page_config(page_title="Production Agent", layout="wide")

st.markdown("""
<style>
    .step-box { border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    .step-done { background-color: #e8f5e9; border-color: #a5d6a7; }
    .step-pending { background-color: #f5f5f5; }
    .stExpander { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("Structured AI Agent")

# Sidebar for configuration and info
with st.sidebar:
    st.header("Agent Configuration")
    st.info("This agent uses a Plan-Execute-Reflect architecture.")

    st.markdown("### Architecture")
    st.markdown("- **Planner**: Decomposes requests")
    st.markdown("- **Executor**: Runs tools (Tavily, Wikipedia, Calculator)")
    st.markdown("- **Reflector**: Critiques results")
    st.markdown("- **Synthesizer**: Generates final answer")

    if st.checkbox("Show Debug State"):
        st.subheader("Current Agent State")
        state_placeholder = st.empty()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "process":
        # Render historical process logs if we want to keep them (optional)
        # For now, we only show final answers in history to keep it clean,
        # but the current run shows process.
        pass

if prompt := st.chat_input("Enter a complex request (e.g., 'Research Nvidia competitors and calculate their avg PE ratio')"):
    
    # 1. Add User Input to State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the Graph
    with st.chat_message("assistant"):

        # Create a container for the process visualization
        process_container = st.container()

        with process_container:
            status_container = st.status("Agent is working...", expanded=True)
        
        inputs = {"messages": [HumanMessage(content=prompt)]}
        final_answer = ""
        
        # Stream events
        try:
            for event in app.stream(inputs):
                
                # --- VISUALIZE PLANNING ---
                if "planner" in event:
                    plan: PlanState = event["planner"]["plan"]
                    status_container.write(f"📝 **Plan Created:** {plan.rationale}")
                    with status_container.expander("View Plan Details"):
                        for step in plan.steps:
                            icon = "🛠️" if step.tool_required else "🧠"
                            st.markdown(f"**{step.step_id}.** {icon} {step.description}")
                
                # --- VISUALIZE EXECUTION ---
                if "executor" in event:
                    # Check for batch execution
                    if "executed_indices" in event["executor"]:
                        indices = event["executor"]["executed_indices"]
                    else:
                        # Fallback for compatibility or single step
                        idx = event["executor"]["current_step"] - 1
                        indices = [idx]

                    for idx in indices:
                        result = event["executor"]["step_results"][idx]
                        status_container.write(f"⚙️ **Step {idx + 1} Executed**")
                        with status_container.expander(f"Result for Step {idx+1}"):
                            st.code(result)

                # --- VISUALIZE REFLECTION ---
                if "reflector" in event:
                    reflection: ReflectorState = event["reflector"]["reflection"]
                    if reflection.is_satisfactory:
                        status_container.write("✅ **Quality Check Passed**")
                    else:
                        status_container.write(f"🤔 **Critique:** {reflection.feedback}")

                # --- VISUALIZE FINAL OUTPUT ---
                if "synthesizer" in event:
                    final_answer = event["synthesizer"]["final_output"]

            # Close the status container
            status_container.update(label="Processing Complete", state="complete", expanded=False)

            # 3. Final Output
            st.markdown("### Final Response")
            st.markdown(final_answer)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            status_container.update(label="Error Occurred", state="error")
            st.error(f"An error occurred: {e}")

