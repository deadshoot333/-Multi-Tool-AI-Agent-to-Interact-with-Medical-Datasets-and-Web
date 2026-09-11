"""
DiabetesDBTool

Exposes diabetes.db as a function tool for the OpenAI Agents SDK. Internally
delegates SQL generation/execution to a LangChain SQL agent (see
sql_tool_base.py) scoped only to this database.
"""

import os

from agents import function_tool

from tools.sql_tool_base import SQLDatasetTool

DIABETES_DB_PATH = os.getenv("DIABETES_DB_PATH", "db/diabetes.db")

_diabetes_tool = SQLDatasetTool(
    db_path=DIABETES_DB_PATH,
    dataset_description=(
        "Diabetes Prediction dataset — per-patient records including gender, age, "
        "hypertension, heart disease, smoking history, BMI, HbA1c level, blood "
        "glucose level, and a diabetes column indicating diagnosis."
    ),
)


@function_tool
def diabetes_db_tool(question: str) -> str:
    """
    Answer statistical/data questions about the Diabetes Prediction dataset
    by querying diabetes.db. Use this for questions involving counts,
    averages, comparisons, filters, or trends in the diabetes patient
    records (e.g. "What's the average HbA1c level for diabetic patients?",
    "How many patients under 30 have hypertension?").

    Args:
        question: The user's natural-language question about the dataset.
    """
    return _diabetes_tool.query(question)
