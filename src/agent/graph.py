from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import (
    node_planner,
    node_executor,
    node_reflector,
    node_synthesizer,
    execution_router,
    reflection_router
)

# Graph Definition
graph = StateGraph(AgentState)

# Nodes
graph.add_node("planner", node_planner)
graph.add_node("executor", node_executor)
graph.add_node("reflector", node_reflector)
graph.add_node("synthesizer", node_synthesizer)

# Edges
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", execution_router, {"executor": "executor", "reflector": "reflector"})
graph.add_conditional_edges("reflector", reflection_router, {"planner": "planner", "synthesizer": "synthesizer"})
graph.add_edge("synthesizer", END)

# Compile
app = graph.compile()
