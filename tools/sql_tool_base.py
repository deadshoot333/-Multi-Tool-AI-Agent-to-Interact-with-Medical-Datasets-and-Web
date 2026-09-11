"""
sql_tool_base.py

Shared helper that wraps a single SQLite database with a LangChain SQL
agent. Each DB-specific tool (HeartDiseaseDBTool, CancerDBTool,
DiabetesDBTool) creates one instance of this class pointed at its own
database, so each dataset stays isolated — the heart disease tool can only
see heart_disease.db, etc.

Uses LangChain's SQLDatabase + create_sql_agent (a ReAct-style agent that
can write and run SQL, inspect the schema, and self-correct on errors)
under the hood. The result is converted back to plain natural language
before being returned to the OpenAI Agents SDK layer.
"""

import os

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI


class SQLDatasetTool:
    def __init__(self, db_path: str, dataset_description: str, model: str = "gpt-4o-mini"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found at {db_path}. "
                f"Run data_prep/csv_to_sqlite.py first."
            )

        self.db_path = db_path
        self.dataset_description = dataset_description

        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.llm = ChatOpenAI(model=model, temperature=0)

        # verbose=False keeps stdout clean; set True while debugging to see
        # the agent's intermediate SQL and reasoning steps.
        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=False,
        )

    def query(self, question: str) -> str:
        """
        Run a natural-language question against this dataset and return a
        natural-language answer. The LangChain SQL agent handles writing
        the SQL, executing it, and summarizing the result.
        """
        prompt = (
            f"You are answering questions about this dataset: {self.dataset_description}\n"
            f"Answer the following question using SQL queries against the database. "
            f"Give a clear, concise natural-language answer (include the key numbers), "
            f"not raw SQL output.\n\nQuestion: {question}"
        )
        try:
            result = self.agent.invoke({"input": prompt})
            return result.get("output", str(result))
        except Exception as exc:  # noqa: BLE001
            return (
                f"I couldn't complete that query against the dataset "
                f"({self.dataset_description}). Error: {exc}"
            )
