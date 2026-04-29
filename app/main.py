import os
import logging
import warnings
from dotenv import load_dotenv

# Suppress dependency warnings
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", message="urllib3 .* doesn't match a supported version!")

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.graph import build_graph

def setup_logging():
    log_format = "  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*60)
    print("  RAGWORKS - Autonomous Research & Report Generation")
    print("  Powered by MCP + Multi-Agent Architecture")
    print("="*60)
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("  [Observability] LangSmith Tracing: ENABLED [ON]")
    else:
        print("  [Observability] LangSmith Tracing: DISABLED [OFF]")
    print("="*60 + "\n")

def main():
    setup_logging()
    logger = logging.getLogger("ragworks.main")
    print_header()
    
    query = input("Enter your research query (or 'quit' to exit): ").strip()
    if query.lower() in ['quit', 'exit', 'q']:
        return

    app = build_graph()
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
        "status": ""
    }

    logger.info("Starting research pipeline...")
    final_state = app.invoke(initial_state)

    def safe(text):
        if isinstance(text, str):
            return text.encode("ascii", "replace").decode("ascii")
        return str(text)

    print("\n" + "=" * 60)
    print("  FINAL RESEARCH REPORT")
    print("=" * 60)
    print(safe(final_state["report"]))

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Original Query:    {safe(final_state['original_query'])}")
    print(f"  Research Query:    {safe(final_state['research_query'])}")
    print(f"  Raw Sources Found: {len(final_state['sources'])}")
    print(f"  Curated Sources:   {len(final_state.get('curated_sources', []))}")
    print(f"  Review Cycles:     {final_state['review_count']}")
    print(f"  Final Status:      {safe(final_state['status'])}")

if __name__ == "__main__":
    main()
