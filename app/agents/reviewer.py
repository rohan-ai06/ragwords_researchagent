import os
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

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


    summaries = ""
    for i, r in enumerate(results):
        summaries += f"\nResult {i+1} [{r.get('source', 'tool')}]: {r.get('title', 'Untitled')}\n  {str(r.get('content') or r.get('summary', ''))[:300]}\n"

    prompt = f"""You are a Research Quality Reviewer.

YOUR TASKS:
1. Look at each result below and decide if it is high-quality and relevant.
2. Put the 1-based index numbers of good results into "relevant_indices".
3. If fewer than {MIN_RELEVANT_SOURCES} results are good, fill "retry_plan" with new search steps.
   CRITICAL: Analyze WHY the current results failed (e.g., too generic, wrong context).
   Based on that feedback, write completely NEW, highly targeted queries that will succeed.
   Each step needs a "query" string and a "tool" string.
   Valid tool names are: search_web, search_arxiv, search_wikipedia, search_semantic_scholar

RESULTS TO EVALUATE:
{summaries}

QUERIES ALREADY TRIED (do not repeat these or anything similar): {tried_queries}

Respond using the ReviewerOutput schema."""

    structured_llm = llm.with_structured_output(ReviewerOutput)
    
    max_validation_retries = 3
    result = None
    
    for attempt in range(max_validation_retries):
        try:
            result = structured_llm.invoke(prompt).model_dump()
            break
        except Exception as e:
            logger.warning(f"Reviewer guardrail (Attempt {attempt+1}/{max_validation_retries}): {e}")
            
    if not result:
        logger.error("Reviewer schema validation failed completely after retries.")
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
    
    # Verdict Logic
    if len(prev_curated) >= MIN_RELEVANT_SOURCES:
        verdict = "approved"
    elif retry_plan:
        verdict = "retry"
    else:
        # If no retry plan was given, but we still have 0 sources, we MUST force a retry
        if len(prev_curated) == 0:
            logger.warning("0 sources approved and no retry plan provided. Forcing fallback retry.")
            verdict = "retry"
            fallback_query = state.get("research_query", "") or state.get("original_query", "general information")
            retry_plan = [{"query": fallback_query, "tool": "search_web"}]
        else:
            verdict = "approved"

    logger.info(f"  Curated: {len(prev_curated)} | Verdict: {verdict.upper()}")
    return {
        "review_verdict": verdict, "review_count": review_count, "retry_plan": retry_plan if verdict == "retry" else [],
        "curated_results": prev_curated, "curated_sources": prev_curated_src,
        "status": f"Review #{review_count}: {verdict}"
    }
