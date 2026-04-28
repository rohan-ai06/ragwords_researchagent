import os
import json
import uuid
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.graph import build_graph

# Setup logging
logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger("ragworks.web")

app = FastAPI(title="RAGWORKS Research Dashboard")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active jobs in memory
jobs = {}

@app.post("/api/research")
async def start_research(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        return {"error": "Query is required"}
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"query": query, "status": "queued"}
    return {"job_id": job_id}

@app.get("/api/stream/{job_id}")
async def stream_research(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}

    query = jobs[job_id]["query"]
    graph = build_graph()

    async def event_generator():
        initial_state = {
            "original_query": query,
            "research_query": "",
            "sub_queries": [],
            "research_plan": [],
            "needs_clarification": False,
            "clarification_question": "",
            "search_results": [],
            "sources": [],
            "review_verdict": "",
            "retry_plan": [],
            "review_count": 0,
            "tried_queries": [],
            "curated_results": [],
            "curated_sources": [],
            "report": "",
            "status": "Starting..."
        }

        try:
            # We use stream_mode="updates" to catch each node's output
            async for event in graph.astream(initial_state, stream_mode="updates"):
                # Clean up the event data for JSON serialization
                serializable_event = {}
                node_name = list(event.keys())[0]
                node_data = event[node_name]
                
                # Extract relevant fields to send to UI
                serializable_event = {
                    "node": node_name,
                    "status": node_data.get("status", ""),
                    "research_query": node_data.get("research_query", ""),
                    "research_plan": node_data.get("research_plan", []),
                    "curated_count": len(node_data.get("curated_results", [])),
                    "review_count": node_data.get("review_count", 0),
                    "report": node_data.get("report", ""),
                    "sources": node_data.get("curated_sources", []) or node_data.get("sources", [])
                }
                
                yield f"data: {json.dumps(serializable_event)}\n\n"
                await asyncio.sleep(0.1)

            yield "data: {\"done\": true}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Serve static files
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
