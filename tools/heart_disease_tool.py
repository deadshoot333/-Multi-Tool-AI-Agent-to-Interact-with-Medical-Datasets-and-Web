"""
HeartDiseaseDBTool

Exposes heart_disease.db as a function tool for the OpenAI Agents SDK.
Internally delegates SQL generation/execution to a LangChain SQL agent
(see sql_tool_base.py) scoped only to this database.
"""

import os

from agents import function_tool

from tools.sql_tool_base import SQLDatasetTool

HEART_DB_PATH = os.getenv("HEART_DISEASE_DB_PATH", "db/heart_disease.db")

_heart_tool = SQLDatasetTool(
    db_path=HEART_DB_PATH,
    dataset_description=(
        "Heart Disease dataset — per-patient clinical records including age, sex, "
        "chest pain type, resting blood pressure, cholesterol, fasting blood sugar, "
        "resting ECG results, max heart rate achieved, exercise-induced angina, "
        "ST depression, and a target column indicating presence of heart disease."
    ),
)


@function_tool
def heart_disease_db_tool(question: str) -> str:
    """
    Answer statistical/data questions about the Heart Disease dataset by
    querying heart_disease.db. Use this for questions involving counts,
    averages, comparisons, filters, or trends in the heart disease patient
    records (e.g. "What's the average cholesterol for patients over 50 with
    heart disease?", "How many male patients have exercise-induced angina?").

    Args:
        question: The user's natural-language question about the dataset.
    """
    return _heart_tool.query(question)
