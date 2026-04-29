import os
import json
import requests
import wikipedia
import arxiv
from dotenv import load_dotenv
from fastmcp import FastMCP
from tavily import TavilyClient

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

mcp = FastMCP("RAGWORKS Research Tools")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
arxiv_client = arxiv.Client()

@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """Search live web via Tavily. Best for recent news and general info."""
    response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
    results = [{"source": "Web", "title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")} for r in response.get("results", [])]
    return json.dumps(results, indent=2)

@mcp.tool()
def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for technical academic papers."""
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = [{"source": "arXiv", "title": r.title, "authors": [a.name for a in r.authors], "summary": r.summary, "url": str(r.pdf_url)} for r in arxiv_client.results(search)]
    return json.dumps(results, indent=2)

@mcp.tool()
def search_wikipedia(query: str) -> str:
    """Lookup facts and verified definitions from Wikipedia."""
    try:
        results = wikipedia.search(query)
        if not results: return json.dumps([])
        page = wikipedia.page(results[0], auto_suggest=False)
        return json.dumps([{"source": "Wikipedia", "title": page.title, "url": page.url, "content": page.summary}], indent=2)
    except: return json.dumps([])

@mcp.tool()
def search_semantic_scholar(query: str, max_results: int = 3) -> str:
    """Search peer-reviewed papers across all sciences."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,abstract,url,citationCount,year"
    try:
        data = requests.get(url).json()
        results = [{"source": "SemanticScholar", "title": p.get("title", ""), "year": p.get("year"), "citations": p.get("citationCount"), "url": p.get("url"), "content": p.get("abstract") or "No abstract."} for p in data.get("data", [])]
        return json.dumps(results, indent=2)
    except: return json.dumps([])

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)
