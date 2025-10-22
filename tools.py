import pandas as pd
import requests
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import uuid
from pathlib import Path
from datetime import datetime
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import os, json

# --- Load CSV for analysis ---
def load_data_agent():
    df = pd.read_csv("data.csv")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    # WARNING: allow_dangerous_code=True enables executing python in a REPL
    # which can be dangerous. This project uses it to allow dataframe analysis.
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=False, allow_dangerous_code=True)

# --- Tool for retrieving data from the CSV file ---
def search_data_tool(query: str) -> str:
    """Analyze the historical CSV data using an internal pandas agent.

    The tool accepts a single text `query` describing the analysis to run and
    returns the agent's text output. This avoids requiring the agent framework
    to pass a DataFrame argument into the Tool call.
    """
    try:
        agent = load_data_agent()
        response = agent.invoke(query)
        # different agent implementations may return a dict with 'output'
        if isinstance(response, dict):
            return response.get("output") or str(response)
        return str(response)
    except Exception as e:
        return f"Data analysis error: {e}"

# --- PubMed search tool ---
def search_pubmed(query: str) -> str:
    """Fetch PubMed summaries for relevant studies."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}
    res = requests.get(url, params=params)
    ids = res.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return "No PubMed articles found."

    summaries = []
    for pmid in ids:
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {"db": "pubmed", "id": pmid, "retmode": "json"}
        summary_res = requests.get(summary_url, params=summary_params)
        data = summary_res.json()
        doc = data.get("result", {}).get(pmid, {})
        summaries.append(f"- {doc.get('title', 'No title')} ({pmid})")
    return "\n".join(summaries)

def save_to_file(content: str, filename: str | None = None):
    os.makedirs("outputs", exist_ok=True)
    if filename:
        filepath = f"outputs/{filename}"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"outputs/prediction_{ts}_{uuid.uuid4().hex[:8]}.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    # return the file path so callers can inspect the saved file
    return filepath

class SaveInput(BaseModel):
    content: str = Field(..., description="The JSON output to save.")
    filename: str | None = Field(None, description="Optional filename.")

# --- Convert to LangChain tools ---
data_tool = Tool(
    name="clinical_trial_data_analysis",
    func=search_data_tool,
    description="Analyze historical clinical trial data for outcome trends and intervention efficacy."
)

pubmed_tool = Tool(
    name="pubmed_search",
    func=search_pubmed,
    description="Search PubMed for related trials and summarize findings."
)

save_tool = StructuredTool.from_function(
    func=save_to_file,
    args_schema=SaveInput,
    name="save_tool",
    description="Save JSON output to a text file inside outputs/."
)


tools = [data_tool, pubmed_tool, save_tool]