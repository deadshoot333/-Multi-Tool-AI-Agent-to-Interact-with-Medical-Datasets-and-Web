# Multi-Tool AI Agent for Medical Datasets

A multi-tool AI agent that answers questions about three medical datasets
(Heart Disease, Cancer, Diabetes) via SQL, and answers general medical
knowledge questions (definitions, symptoms, cures) via web search.

- **Main agent**: built with the **OpenAI Agents SDK** — routes each
  question to the right tool.
- **DB tools**: each one wraps a **LangChain SQL agent** scoped to a single
  SQLite database, so the agent can write and run SQL against that dataset
  and summarize the result in plain language.
- **Web search tool**: uses **Tavily** for general medical knowledge
  lookups.

## Project structure

```
medical-agent/
├── data/                     # put downloaded Kaggle CSVs here
├── db/                       # generated SQLite databases go here
├── data_prep/
│   └── csv_to_sqlite.py      # converts CSVs -> SQLite DBs
├── tools/
│   ├── sql_tool_base.py      # shared LangChain SQL agent wrapper
│   ├── heart_disease_tool.py # HeartDiseaseDBTool
│   ├── cancer_tool.py        # CancerDBTool
│   ├── diabetes_tool.py      # DiabetesDBTool
│   └── web_search_tool.py    # MedicalWebSearchTool
├── agent/
│   └── main_agent.py         # main routing agent + CLI
├── requirements.txt
├── .env.example
└── README.md
```

## 1. Setup

```bash
git clone <your-repo-url> medical-agent   # or just use this folder
cd medical-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

- Get an OpenAI key at https://platform.openai.com/api-keys
- Get a free Tavily key at https://tavily.com (used for
  `MedicalWebSearchTool` instead of SerpAPI/Bing — swap in
  `tools/web_search_tool.py` if you'd rather use one of those).

## 2. Download the datasets

Download these three CSVs from Kaggle (you'll need a free Kaggle account
and `kaggle.json` API credentials, or just download manually from the
website):

| Dataset | Link | Save as |
|---|---|---|
| Heart Disease | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset | `data/heart.csv` |
| Cancer Prediction | https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset | `data/cancer.csv` |
| Diabetes Prediction | https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset | `data/diabetes.csv` |

Using the Kaggle CLI instead:

```bash
pip install kaggle
kaggle datasets download -d johnsmith88/heart-disease-dataset -p data --unzip
kaggle datasets download -d rabieelkharoua/cancer-prediction-dataset -p data --unzip
kaggle datasets download -d iammustafatz/diabetes-prediction-dataset -p data --unzip
```

Then rename whatever CSV filename each dataset unzips to, to match
`data/heart.csv`, `data/cancer.csv`, `data/diabetes.csv` (Kaggle dataset
filenames vary/change over time — check the unzipped contents).

## 3. Convert CSVs to SQLite

```bash
python data_prep/csv_to_sqlite.py
```

This creates `db/heart_disease.db`, `db/cancer.db`, and `db/diabetes.db`,
each with one properly-typed table (`heart_disease`,
`cancer_patients`, `diabetes_patients` respectively).

## 4. Run the agent

Interactive CLI:

```bash
python agent/main_agent.py
```

Example session:

```
You: How many patients in the heart disease dataset have a resting blood pressure above 140?
Assistant: 187 patients have a resting blood pressure above 140 mmHg.

You: What are the common symptoms of type 2 diabetes?
Assistant: Common symptoms include increased thirst, frequent urination,
fatigue, blurred vision, slow-healing sores, and unexplained weight loss...

You: What's the average BMI of cancer patients with a family history of cancer, and what are the main risk factors for cancer?
Assistant: [combines cancer_db_tool + medical_web_search_tool output]
```

Or call it from Python:

```python
import asyncio
from agent.main_agent import run_query

answer = asyncio.run(run_query("Average glucose level for diabetic patients?"))
print(answer)
```

## How routing works

The main agent (`agent/main_agent.py`) is given all four tools and
instructions telling it:

- **Data/statistics questions** about a specific dataset → the matching
  `*_db_tool`, which runs a LangChain SQL agent against that dataset's
  SQLite DB and returns a natural-language answer.
- **General medical knowledge** (definitions, symptoms, causes, cures) →
  `medical_web_search_tool`, which queries Tavily and returns a summarized
  answer with sources.
- **Mixed questions** → the agent calls both tools and combines the
  results.

Each DB tool only has access to its own database — `heart_disease_db_tool`
cannot see cancer or diabetes data, and vice versa — so routing mistakes
fail safely rather than silently querying the wrong dataset.

## Notes / things to double-check before submitting

- Column names get normalized to snake_case during CSV→SQLite conversion
  (e.g. `Blood Pressure` → `blood_pressure`) — check `db/*.db` with a
  SQLite browser if you want to confirm the exact schema the SQL agents
  see.
- `create_sql_agent(..., verbose=False)` in `tools/sql_tool_base.py` —
  set `verbose=True` while developing to see the generated SQL and
  reasoning steps in the console.
- The Kaggle dataset download links may occasionally change their exact
  CSV filename inside the zip; rename to the expected filenames in
  `data/` as noted above.
- Swap `medical_web_search_tool`'s Tavily call for SerpAPI or Bing by
  editing `tools/web_search_tool.py` if your course specifically requires
  one of those.
