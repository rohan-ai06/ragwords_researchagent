import os
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from app.mcp_client_utils import get_mcp_tools_list_sync

logger = logging.getLogger("ragworks.intake")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

class ResearchPlanStep(BaseModel):
    query: str = Field(description="Search query optimized for the chosen tool.")
    tool: str = Field(description="The exact name of the tool to use.")

class IntakeOutput(BaseModel):
    research_query: str = Field(description="Rewritten professional query", default="")
    research_plan: List[ResearchPlanStep] = Field(description="Strategic search plan", default_factory=list)
    needs_clarification: bool = Field(description="True if query is vague", default=False)
    clarification_question: str = Field(description="Question for user if clarification needed", default="")

def intake_node(state: dict) -> dict:
    """Processes user query and generates a strategic research plan."""
    raw_query = state.get("original_query", "")
    logger.info("=" * 50)
    logger.info("INTAKE AGENT — Strategic Planning")
    logger.info(f"Input: {raw_query}")

    tools_list_string = get_mcp_tools_list_sync()

    prompt = f"""You are a Strategic Research Query Specialist. 
1. REWRITE it into a clear, professional research query.
2. Formulate a SURGICAL RESEARCH PLAN using the toolkit below.

AVAILABLE TOOLKIT:
{tools_list_string}

PLAN RULES:
- For EACH tool you select, generate 2-3 distinct search queries.
- DO NOT USE ALL TOOLS. Pick only the most relevant ones.
- If needs_clarification is true, you must still provide a best-guess research_query.

User's input: "{raw_query}" """

    structured_llm = llm.with_structured_output(IntakeOutput)
    max_validation_retries = 3
    result = None

    for attempt in range(max_validation_retries):
        try:
            result_obj = structured_llm.invoke(prompt)
            result = result_obj.model_dump()
            break
        except Exception as e:
            logger.warning(f"Guardrail retry {attempt+1}: {e}")

    if not result:
        return {
            "research_query": raw_query,
            "research_plan": [],
            "status": "HALTED: Guardrail validation failed."
        }

    plan = result.get("research_plan", [])
    all_sub = [step["query"] for step in plan]

    for step in plan:
        logger.info(f"  [{step['tool']}] -> {step['query']}")

    return {
        "research_query": result["research_query"],
        "sub_queries": all_sub,
        "research_plan": plan,
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question", ""),
        "status": f"Plan created with {len(plan)} steps."
    }

def human_clarification_node(state: dict) -> dict:
    """Handles human interaction for vague queries."""
    question = state.get("clarification_question", "Could you provide more details?")
    print(f"\n[Human-in-the-Loop] -> {question}")
    user_response = input("\nYour response: ").strip()
    return {
        "original_query": f"{state['original_query']} -- Context: {user_response}",
        "needs_clarification": False,
        "status": "Clarification received."
    }
