"""
CancerDBTool

Exposes cancer.db as a function tool for the OpenAI Agents SDK. Internally
delegates SQL generation/execution to a LangChain SQL agent (see
sql_tool_base.py) scoped only to this database.
"""

import os

from agents import function_tool

from tools.sql_tool_base import SQLDatasetTool

CANCER_DB_PATH = os.getenv("CANCER_DB_PATH", "db/cancer.db")

_cancer_tool = SQLDatasetTool(
    db_path=CANCER_DB_PATH,
    dataset_description=(
        "Cancer Prediction dataset — per-patient records including age, gender, "
        "BMI, smoking status, genetic risk, physical activity, alcohol intake, "
        "cancer history, and a diagnosis column indicating cancer presence."
    ),
)


@function_tool
def cancer_db_tool(question: str) -> str:
    """
    Answer statistical/data questions about the Cancer Prediction dataset by
    querying cancer.db. Use this for questions involving counts, averages,
    comparisons, filters, or trends in the cancer patient records (e.g.
    "What percentage of smokers were diagnosed with cancer?", "Average BMI
    of patients with a family cancer history?").

    Args:
        question: The user's natural-language question about the dataset.
    """
    return _cancer_tool.query(question)
