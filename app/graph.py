from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.agents.intake import intake_node, human_clarification_node
from app.agents.researcher import researcher_node
from app.agents.reviewer import reviewer_node
from app.agents.writer import writer_node

def route_after_intake(state: AgentState) -> str:
    return "clarify" if state.get("needs_clarification") else "researcher"

def route_after_review(state: AgentState) -> str:
    return "researcher" if state.get("review_verdict") == "retry" else "writer"

def build_graph():
    """Compiles the LangGraph orchestration for the RAG pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("clarify", human_clarification_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("intake")

    graph.add_conditional_edges("intake", route_after_intake, {
        "clarify": "clarify", "researcher": "researcher"
    })
    graph.add_edge("clarify", "intake")
    graph.add_edge("researcher", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review, {
        "researcher": "researcher", "writer": "writer"
    })
    graph.add_edge("writer", END)

    return graph.compile()
