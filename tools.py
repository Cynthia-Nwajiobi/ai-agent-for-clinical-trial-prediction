import pandas as pd
import requests
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- Load CSV for analysis ---
def load_data_agent():
    df = pd.read_csv("data.csv")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=False)

def analyze_trial_data(query: str) -> str:
    """Analyze historical trial data from CSV."""
    try:
        agent = load_data_agent()
        response = agent.invoke(query)
        return response.get("output", "No output from data agent.")
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

# --- Convert both to LangChain tools ---
data_tool = Tool(
    name="Clinical Trial Data Analysis",
    func=analyze_trial_data,
    description="Analyze historical clinical trial data for outcome trends and intervention efficacy."
)

pubmed_tool = Tool(
    name="PubMed Search",
    func=search_pubmed,
    description="Search PubMed for related trials and summarize findings."
)

tools = [data_tool, pubmed_tool]
