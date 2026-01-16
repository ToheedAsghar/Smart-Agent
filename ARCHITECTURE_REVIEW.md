# Architecture Review: Structured AI Agent

## Current Architecture
The current implementation uses a **Plan-Execute-Reflect** pattern built with LangGraph.

### Flow
1.  **Planner**: Decomposes the user query into a sequential list of steps.
2.  **Executor**: Executes steps one by one. If a step requires tools, it uses them; otherwise, it reasons using the LLM.
3.  **Reflector**: After execution, it evaluates if the results satisfy the user query.
    - If **Satisfactory**: Proceeds to Synthesis.
    - If **Not Satisfactory**: Loops back to the Planner with feedback to generate a new plan (up to 3 retries).
4.  **Synthesizer**: Compiles all gathered information into a final response.

### Pros
-   **Reliability**: Planning helps in breaking down complex tasks.
-   **Quality Control**: Reflection ensures the answer is verified before being shown.
-   **Structure**: Clear separation of concerns between planning, execution, and review.

### Cons
-   **Latency**: The loop (Plan -> Execute -> Reflect -> Retry) can be slow.
-   **Rigidity**: The plan is static. If step 1 reveals step 2 is unnecessary or needs changing, the agent has to wait until all steps are done to reflect and replan.
-   **Context Limit**: Passing full message history and step results can grow large.

---

## Recommendations for Enterprise Production

### 1. Dynamic Replanning (Reactive Planning)
Instead of executing the entire plan blindly, the agent should be able to update the plan after *each* step.
-   **Why**: If step 1 fails or provides unexpected info, the rest of the plan might be invalid.
-   **How**: After `node_executor` completes a step, route back to `node_planner` (or a lightweight "updater" node) to adjust remaining steps.

### 2. Parallel Execution
Steps that don't depend on each other should run in parallel.
-   **How**: Use LangGraph's support for parallel node execution (fan-out/fan-in) for independent research tasks.

### 3. Long-Term Memory (checkpointing)
Currently, state is transient.
-   **Why**: To support multi-turn conversations where the user refers back to previous requests.
-   **How**: Use `langgraph-checkpoint` with a persistent backend (Postgres, Redis) to save thread state.

### 4. Human-in-the-Loop
For sensitive tasks, add a human review node.
-   **How**: Use `interrupt_before` in LangGraph to pause before the `executor` or `synthesizer` runs, allowing a human to approve the plan or the final answer.

### 5. Hierarchical Agents
For very complex domains, use a Supervisor-Worker pattern.
-   **Structure**: A "Supervisor" agent delegates sub-tasks to specialized "Worker" agents (e.g., a "Coder" agent, a "Researcher" agent).

## Conclusion
The current structure is a solid foundation for "Reasoning" agents. The Refactoring performed in this PR (splitting into `src/agent`, `src/tools`) moves it towards a maintainable enterprise application. The next logical step would be adding **Checkpointers** for persistence and **Dynamic Replanning** for efficiency.
