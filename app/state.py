from typing import List, Dict
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """Shared state across all agents in the RAGWORKS pipeline."""
    # Intake
    original_query: str
    research_query: str
    sub_queries: List[str]
    research_plan: List[Dict]
    needs_clarification: bool
    clarification_question: str

    # Researcher
    search_results: List[Dict]
    sources: List[Dict]

    # Reviewer
    review_verdict: str
    retry_plan: List[Dict]
    review_count: int
    tried_queries: List[str]
    curated_results: List[Dict]
    curated_sources: List[Dict]

    # Writer
    report: str

    # Tracking
    status: str
