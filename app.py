import streamlit as st
import json
from langchain_core.messages import HumanMessage
from backend import app, PlanState, ReflectorState# Import our types

st.set_page_config(page_title="Production Agent", layout="wide")

st.markdown("""
<style>
    .step-box { border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    .step-done { background-color: #e8f5e9; border-color: #a5d6a7; }
    .step-pending { background-color: #f5f5f5; }
</style>
""", unsafe_allow_html=True)

st.title("Structured Agent")

# Sidebar for debug/tracing
with st.sidebar:
    st.header("Agent State")
    state_display = st.json({})

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display simple chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter a complex request (e.g., 'Research the competitors of Nvidia and draft a comparison table')"):
    
    # 1. Add User Input to State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the Graph
    with st.chat_message("assistant"):
        status_box = st.status("Initializing Agent...", expanded=True)
        
        inputs = {"messages": [HumanMessage(content=prompt)]}
        final_answer = ""
        
        # We define a placeholder for the Plan UI
        plan_container = st.empty()
        
        for event in app.stream(inputs):
            
            # --- VISUALIZE PLANNING ---
            if "planner" in event:
                plan: PlanState = event["planner"]["plan"]
                status_box.write(f" **Plan Created:** {plan.rationale}")
                
                # Render the plan nicely
                # with plan_container.container():
                #     st.subheader("Execution Plan")
                #     for step in plan.steps:
                #         st.markdown(f"**Step {step.step_id}:** {step.description}")
            
            # --- VISUALIZE EXECUTION ---
            if "executor" in event:
                idx = event["executor"]["current_step"] - 1
                result = event["executor"]["step_results"][idx]
                
                # Update status
                status_box.write(f" **Step {idx + 1} Complete**")
                with status_box.expander(f"View Result for Step {idx+1}"):
                    st.code(result)

            # --- VISUALIZE REFLECTION ---
            if "reflector" in event:
                reflection: ReflectorState = event["reflector"]["reflection"]
                if reflection.is_satisfactory:
                    status_box.update(label="Quality Check Passed", state="complete", expanded=False)
                else:
                    status_box.write(f" **Critique:** {reflection.feedback}")

            # --- VISUALIZE FINAL OUTPUT ---
            if "synthesizer" in event:
                final_answer = event["synthesizer"]["final_output"]

        # 3. Final Output
        st.markdown("### Final Response")
        st.markdown(final_answer)
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer})