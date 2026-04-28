import os
import json
import asyncio
import logging
from fastmcp import Client

logger = logging.getLogger("ragworks.researcher")
MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mcp", "mcp_server.py")

async def _execute_plan_step(client, step: dict) -> tuple[list[dict], list[dict]]:
    query, tool = step.get("query", ""), step.get("tool", "")
    results, sources = [], []
    try:
        response = await client.call_tool(tool, {"query": query})
        if response and hasattr(response, 'is_error') and response.is_error:
             return [], []
        if response and response.content:
            data = json.loads(response.content[0].text)
            for item in data:
                results.append(item)
                sources.append({"title": item.get("title", "Unknown"), "url": item.get("url", ""), "source_type": item.get("source", tool)})
    except Exception as e:
        logger.error(f"[{tool}] error: {e}")
    return results, sources

async def _run_dynamic_search(plan: list) -> tuple[list[dict], list[dict]]:
    all_results, all_sources = [], []
    
    # Split plan into Parallel (all tools except arXiv) and Sequential (arXiv)
    parallel_steps = [s for s in plan if s.get("tool") != "search_arxiv"]
    sequential_steps = [s for s in plan if s.get("tool") == "search_arxiv"]

    async with Client(MCP_SERVER_PATH) as client:
        # 1. Execute Parallel Block
        if parallel_steps:
            logger.info(f"  Firing {len(parallel_steps)} parallel search tasks...")
            tasks = [_execute_plan_step(client, step) for step in parallel_steps]
            parallel_outputs = await asyncio.gather(*tasks)
            for results, sources in parallel_outputs:
                all_results.extend(results)
                all_sources.extend(sources)

        # 2. Execute Sequential Block (arXiv)
        if sequential_steps:
            logger.info(f"  Executing {len(sequential_steps)} arXiv tasks sequentially...")
            for step in sequential_steps:
                results, sources = await _execute_plan_step(client, step)
                all_results.extend(results)
                all_sources.extend(sources)
                await asyncio.sleep(1.0) # Be polite to arXiv

    return all_results, all_sources

def researcher_node(state: dict) -> dict:
    review_count = state.get("review_count", 0)
    plan = state.get("retry_plan" if review_count > 0 else "research_plan", [])
    
    logger.info("=" * 50)
    logger.info(f"RESEARCHER AGENT — {'Retry cycle #' + str(review_count) if review_count > 0 else 'Initial Search'}")
    
    results, sources = asyncio.run(_run_dynamic_search(plan))
    
    all_results = state.get("search_results", []) + results
    all_sources = state.get("sources", []) + sources
    all_tried = state.get("tried_queries", []) + [step.get("query") for step in plan]

    return {
        "search_results": all_results, "sources": all_sources, "tried_queries": all_tried,
        "status": f"Research complete. {len(all_results)} results total."
    }
