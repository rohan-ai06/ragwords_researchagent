import os
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from app.mcp_client_utils import get_mcp_tools_list_sync

logger = logging.getLogger("ragworks.reviewer")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

MAX_RETRIES = 3
MIN_RELEVANT_SOURCES = 10

class RetryPlanStep(BaseModel):
    query: str = Field(description="Search query optimized for the chosen tool.")
    tool: str = Field(description="The exact name of the tool to use.")

class ReviewerOutput(BaseModel):
    relevant_indices: List[int] = Field(description="1-based index numbers of relevant results.")
    reason: str = Field(description="Explanation of curation decision")
    missing_aspects: List[str] = Field(description="Uncovered aspects", default_factory=list)
    retry_plan: List[RetryPlanStep] = Field(description="Retry strategic plan", default_factory=list)

def reviewer_node(state: dict) -> dict:
    """Evaluates search results for quality and relevance."""
    review_count = state.get("review_count", 0) + 1
    results, sources = state.get("search_results", []), state.get("sources", [])
    research_plan, tried_queries = state.get("research_plan", []), state.get("tried_queries", [])
    prev_curated, prev_curated_src = state.get("curated_results", []), state.get("curated_sources", [])

    logger.info("=" * 50)
    logger.info(f"REVIEWER AGENT — Review cycle #{review_count}")

    if review_count >= MAX_RETRIES:
        return {
            "review_verdict": "approved", "review_count": review_count, "retry_plan": [],
            "curated_results": prev_curated if prev_curated else results[:15],
            "curated_sources": prev_curated_src if prev_curated_src else sources[:15],
            "status": "Reviewer auto-approved (max retries)."
        }

    tools_list = get_mcp_tools_list_sync()
    summaries = ""
    for i, r in enumerate(results):
        summaries += f"\nResult {i+1} [{r.get('source', 'tool')}]: {r.get('title', 'Untitled')}\n  {str(r.get('content') or r.get('summary', ''))[:300]}\n"

    prompt = f"""You are a Research Quality Reviewer. 
1. GRADE each result for relevance.
2. SELECT only high-quality, non-redundant results.
3. Design a RETRY PLAN if more data is needed (min {MIN_RELEVANT_SOURCES}).

AVAILABLE TOOLKIT: {tools_list}
RESULTS: {summaries}
ALREADY TRIED: {tried_queries} """

    structured_llm = llm.with_structured_output(ReviewerOutput)
    try:
        result = structured_llm.invoke(prompt).model_dump()
    except Exception as e:
        logger.warning(f"Reviewer guardrail: {e}")
        result = {"relevant_indices": [], "reason": "Failed schema", "retry_plan": []}

    curated_res, curated_src = [], []
    for idx in result.get("relevant_indices", []):
        i = idx - 1
        if 0 <= i < len(results):
            curated_res.append(results[i])
            url = results[i].get("url", "")
            match = next((s for s in sources if s.get("url") == url), None)
            if match: curated_src.append(match)

    # Merge unique
    existing_urls = {r.get("url", "") for r in prev_curated}
    for r, s in zip(curated_res, curated_src):
        if r.get("url", "") not in existing_urls:
            prev_curated.append(r)
            prev_curated_src.append(s)
            existing_urls.add(r.get("url", ""))

    retry_plan = result.get("retry_plan", [])
    verdict = "approved" if len(prev_curated) >= MIN_RELEVANT_SOURCES or not retry_plan else "retry"

    logger.info(f"  Curated: {len(prev_curated)} | Verdict: {verdict.upper()}")
    return {
        "review_verdict": verdict, "review_count": review_count, "retry_plan": retry_plan if verdict == "retry" else [],
        "curated_results": prev_curated, "curated_sources": prev_curated_src,
        "status": f"Review #{review_count}: {verdict}"
    }
