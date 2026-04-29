import os
from dotenv import load_dotenv
load_dotenv()
from app.graph import build_graph

def test_pipeline():
    print("Initializing test pipeline...")
    graph = build_graph()
    
    query = "give me a summary about difference between the supervised and unsupervised learning in ml"
    print(f"Testing with query: {query}")
    
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
        "status": "Starting test..."
    }
    
    print("Running graph...")
    final_state = graph.invoke(initial_state)
    
    print("\n" + "="*50)
    print(f"Final Status: {final_state.get('status')}")
    print(f"Curated Sources Used: {len(final_state.get('curated_sources', []))}")
    print("="*50)
    print("Report Preview:")
    print(final_state.get('report', '')[:500] + "...")

if __name__ == "__main__":
    test_pipeline()
