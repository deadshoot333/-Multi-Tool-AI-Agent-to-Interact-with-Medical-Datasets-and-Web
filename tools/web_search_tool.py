"""
MedicalWebSearchTool

Exposes general medical knowledge lookup (definitions, symptoms, causes,
treatments) via Tavily's search API, as a function tool for the OpenAI
Agents SDK. This tool should NOT be used for anything involving the
project's own datasets — only for open-ended medical knowledge questions.
"""

import os

from agents import function_tool
from tavily import TavilyClient

_tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@function_tool
def medical_web_search_tool(query: str) -> str:
    """
    Search the web for general medical knowledge — definitions, symptoms,
    causes, risk factors, treatments/cures, or explanations of medical
    conditions. Use this for questions that are NOT about the project's own
    datasets (e.g. "What are the symptoms of type 2 diabetes?", "What
    causes high cholesterol?", "How is heart disease treated?").

    Args:
        query: The medical knowledge question to search for.
    """
    try:
        response = _tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Web search failed: {exc}"

    parts = []
    if response.get("answer"):
        parts.append(f"Summary: {response['answer']}")

    for result in response.get("results", [])[:3]:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        snippet = content[:300] + ("..." if len(content) > 300 else "")
        parts.append(f"- {title}: {snippet} (source: {url})")

    return "\n".join(parts) if parts else "No relevant results found."
