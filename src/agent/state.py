from typing import Optional, TypedDict, Annotated, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import add_messages

class PlanStep(BaseModel):
    step_id: int = Field(..., description="The unique sequential identifier for the step")
    description: str = Field(..., description="A description of the step to be taken")
    tool_required: bool = Field(..., description="Whether this step requires external search/tools?")
    section_id: int = Field(..., description="The ID of the section this step belongs to. Steps with the same section_id can be executed in parallel.")

class PlanState(BaseModel):
    steps: List[PlanStep] = Field(description="List of the steps in the plan")
    rationale: str = Field(description="the reasoning behind the plan, why it is chosen.")

class ReflectorState(BaseModel):
    is_satisfactory: bool = Field(..., description="Whether the current plan and results answer the user's question?")
    feedback: str = Field(..., description="Critique of what is missing")
    next_step_adjustment: Optional[str] = Field(None, description="How to adjust the next steps if needed")

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]

    plan: Optional[PlanState]
    current_step: int
    step_results: Dict[int, str]
    final_output: Optional[str]
    reflection: Optional[ReflectorState]

    retry_cnt: int
