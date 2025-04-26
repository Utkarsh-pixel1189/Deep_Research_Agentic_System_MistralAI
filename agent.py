from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from langchain_mistralai.chat_models import ChatMistralAI
import os

class AgentState(TypedDict):
    user_input: Optional[str]
    research_data: Optional[dict]
    generated_draft: Optional[str]
    error: Optional[str]

def run_research_agent(query: str) -> dict:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily.search(query=query, include_raw_content=True)

def run_draft_agent(research_data: dict) -> str:
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.7
    )
    prompt = f"Create a detailed report based on:\n{str(research_data)[:4000]}"
    return llm.invoke(prompt).content

def research_node(state: AgentState) -> AgentState:
    try:
        query = state.get("user_input")
        if not query:
            raise ValueError("Research query cannot be empty")
        return {**state, "research_data": run_research_agent(query), "error": None}
    except Exception as e:
        return {**state, "error": f"Research failed: {str(e)}"}

def draft_node(state: AgentState) -> AgentState:
    try:
        if state.get("error"):
            return state
        if not state.get("research_data"):
            raise ValueError("No research data available")
        return {**state, "generated_draft": run_draft_agent(state["research_data"]), "error": None}
    except Exception as e:
        return {**state, "error": f"Drafting failed: {str(e)}"}

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_node)
    workflow.add_node("draft", draft_node)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "draft")
    workflow.add_edge("draft", END)
    return workflow.compile()