from langchain_core.messages import SystemMessage, AIMessage
from src.agent.state import AgentState, PlanState, ReflectorState
from src.utils.config import get_llm, get_reasoner_llm
from src.tools.custom_tools import get_tools

# Initialize models and tools
llm = get_llm()
reasoner_llm = get_reasoner_llm()
structured_reasoner_llm = reasoner_llm.with_structured_output(ReflectorState)
structured_llm = llm.with_structured_output(PlanState)

tools = get_tools()
tool_map = {t.name: t for t in tools}
# Bind all tools
tool_chain = llm.bind_tools(tools)

def node_planner(state: AgentState):
    # Ensure messages exist
    if not state.get('messages'):
        return {}

    question = state['messages'][-1].content

    is_retry = (state.get("retry_cnt", 0) > 0)

    if is_retry and state.get('reflection'):
        feedback = state['reflection'].feedback
        prompt = f"""
            You are fixing a failed plan.
            The previous execution failed.
            Feedback from execution: {feedback}
        """
    else:
        prompt = f"""
            You are a Solution Architect. Break down the user's query into
            logical, sequential steps. Be Precise. If it's a greeting, make a 1-step plan.
        """

    # pass the full message history for context
    # Note: structured_llm.invoke expects a list of messages or a string.
    # We prepend the system message.
    input_messages = [SystemMessage(content=prompt)] + state['messages']
    res = structured_llm.invoke(input_messages)

    return {
        'plan': res,
        'current_step': 0,
        'step_results': {}, # reset step results on new plan
        'retry_cnt': state.get('retry_cnt', -1) + 1,
    }

def node_executor(state: AgentState):
    plan = state['plan']
    idx: int = state['current_step']

    if not plan or idx >= len(plan.steps):
        return state

    curr_step = plan.steps[idx]

    if curr_step.tool_required:
        res = tool_chain.invoke(f"Perform this task: {curr_step.description}")

        if res.tool_calls:
            # Execute the first tool call (simplified for this agent structure)
            tool_call = res.tool_calls[0]
            tool_name = tool_call['name']

            if tool_name in tool_map:
                tool = tool_map[tool_name]
                try:
                    output = tool.invoke(tool_call['args'])
                    res = str(output)
                except Exception as e:
                    res = f"Error executing tool {tool_name}: {e}"
            else:
                res = f"Tool {tool_name} not found."
        else:
            res = res.content

    else:
        # Pure reasoning step
        res = llm.invoke(f"Reason through this: {curr_step.description}").content

    return {
        'step_results': {idx: res},
        'current_step':  idx + 1
    }

def node_reflector(state: AgentState):
    if not state.get('messages'):
        return {}

    user_query = state['messages'][0].content
    plan = state['plan']
    step_results = state['step_results']

    context = "\n".join([f"step {i} : {v}" for i,v in step_results.items()])

    prompt = f"""
        You are an expert reviewer. Given the user's query, the plan made, and the results so far, Strictly evaluate: Did we answer the request? If yes, mark is_satisfactory=True. If no, provide strict feedback on what is missing.

        User Query: {user_query}
        Plan: {plan}
        Executed Steps: {context}
    """

    res = structured_reasoner_llm.invoke(prompt)

    # we append the critique as a message for context in next planning
    return {
        'reflection': res,
        "messages": [AIMessage(content=f"Self-Reflection: {res.feedback}")]
    }

def node_synthesizer(state: AgentState):
    """Combines all the gathered information into a final human-readable answer."""

    step_results = state['step_results']
    if not state.get('messages'):
        user_query = "Unknown"
    else:
        user_query = state['messages'][0].content

    context = "\n".join([f"info found: {v}" for v in step_results.values()])

    prompt = f"""
        You are a helpful assistant. Using the information gathered from various steps, provide a concise and accurate answer to user query: '{user_query}' using this data: \n{context}
    """

    res = llm.invoke(prompt)

    return {
        'final_output': res.content
    }

def execution_router(state: AgentState):
    idx: int = state['current_step']
    plan = state['plan']

    if plan and idx < len(plan.steps):
        return "executor"
    else:
        return "reflector"

def reflection_router(state: AgentState):
    reflection = state.get('reflection')
    retries = state.get('retry_cnt', 0)
    MAX_RETRIES = 3

    if reflection and reflection.is_satisfactory:
        return "synthesizer"
    elif retries < MAX_RETRIES:
        return "planner"
    else:
        # print("Max retries reached. Proceeding to synthesizer.")
        return "synthesizer"
