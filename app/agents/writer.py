import os
import logging
from langchain_groq import ChatGroq

logger = logging.getLogger("ragworks.writer")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def writer_node(state: dict) -> dict:
    """Synthesizes curated research into a final structured report."""
    logger.info("=" * 50)
    logger.info("WRITER AGENT — Generating final report")
    
    orig_query = state.get("original_query", "")
    curated = state.get("curated_results", []) or state.get("search_results", [])[:15]
    curated_src = state.get("curated_sources", []) or state.get("sources", [])[:15]

    context = ""
    for i, r in enumerate(curated[:15]):
        context += f"\n--- Source {i+1}: {r.get('title', 'Untitled')} ---\n"
        context += f"{str(r.get('content') or r.get('summary', ''))[:500]}\n"

    refs = ""
    for i, s in enumerate(curated_src[:15]):
        refs += f"[{i+1}] {s.get('title', 'Untitled')} - {s.get('url', 'N/A')}\n"

    prompt = f"""You are a Professional Research Report Writer.
USER ORIGINAL REQUEST: "{orig_query}"

Analyze the user's intent to decide the best format.
- If they said "short/brief": Write max 200 words.
- If they said "bullet points": Use only bullets.
- Default: Detailed structured paper (Executive Summary, Key Findings, Analysis, References).

RESEARCH MATERIAL:
{context}

REFERENCES:
{refs}

Always use [N] citations and provide a References list at the end."""

    response = llm.invoke(prompt)
    report = response.content
    logger.info(f"Report generated. Length: {len(report)} characters.")

    return {
        "report": report,
        "status": "Final report generated."
    }
