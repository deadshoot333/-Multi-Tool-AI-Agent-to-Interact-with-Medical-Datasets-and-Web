"""
main_agent.py

The main Multi-Tool AI Agent. Built with the OpenAI Agents SDK, it holds
four tools:

    - heart_disease_db_tool   (LangChain SQL agent over heart_disease.db)
    - cancer_db_tool          (LangChain SQL agent over cancer.db)
    - diabetes_db_tool        (LangChain SQL agent over diabetes.db)
    - medical_web_search_tool (Tavily web search)

The agent's instructions tell it how to route:
    - statistics / data / numbers about a specific dataset -> matching DB tool
    - definitions / symptoms / causes / cures / general knowledge -> web search

Run interactively:
    python agent/main_agent.py

Or import `run_query` / `agent` from this module to use elsewhere.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Allow running this file directly (python agent/main_agent.py) by adding
# the project root to sys.path so `tools.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from agents import Agent, Runner  # noqa: E402

from tools.heart_disease_tool import heart_disease_db_tool  # noqa: E402
from tools.cancer_tool import cancer_db_tool  # noqa: E402
from tools.diabetes_tool import diabetes_db_tool  # noqa: E402
from tools.web_search_tool import medical_web_search_tool  # noqa: E402


AGENT_INSTRUCTIONS = """
You are a Medical Multi-Tool Assistant with access to four tools:

1. heart_disease_db_tool   - queries the Heart Disease patient dataset
2. cancer_db_tool          - queries the Cancer Prediction patient dataset
3. diabetes_db_tool        - queries the Diabetes Prediction patient dataset
4. medical_web_search_tool - searches the web for general medical knowledge

ROUTING RULES:
- If the question asks about statistics, counts, averages, correlations,
  or any numeric/data query over one of the three specific datasets, call
  the matching *_db_tool. Only call the tool for the dataset the question
  is actually about — don't guess across datasets.
- If the question asks for a definition, explanation of symptoms, causes,
  risk factors, or treatment/cure of a medical condition in general (not
  tied to the project's own data), call medical_web_search_tool.
- If a question needs both (e.g. "What's the average glucose level of
  diabetic patients in the dataset, and what symptoms should they watch
  for?"), call both tools and combine the answers clearly.
- Always answer in clear, plain natural language. Never show raw SQL or
  raw JSON to the user unless they explicitly ask for it.
- If you're not sure which dataset a data question refers to, ask a brief
  clarifying question instead of guessing.
"""

agent = Agent(
    name="Medical Multi-Tool Assistant",
    instructions=AGENT_INSTRUCTIONS,
    tools=[
        heart_disease_db_tool,
        cancer_db_tool,
        diabetes_db_tool,
        medical_web_search_tool,
    ],
    model="gpt-4o-mini",
)


async def run_query(question: str) -> str:
    result = await Runner.run(agent, question)
    return result.final_output


async def _interactive_loop():
    print("Medical Multi-Tool Assistant — type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = await run_query(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    asyncio.run(_interactive_loop())
