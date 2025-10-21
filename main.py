from dotenv import load_dotenv 
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent   
from tools import data_tool, pubmed_tool  

load_dotenv()

class ClinicalTrialResponse(BaseModel):
    endpoint: str
    justification: str
    indication: list[str]
    references: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
parser = PydanticOutputParser(pydantic_object=ClinicalTrialResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ( 
            "system", 
            """
            You are a clinical trial research scientist with 20 years of experience specializing in longevity/anti-aging clinical trials, especially cancer research.
            Your job is: given a trial summary, predict whether it will meet its primary endpoint, using only historical evidence from before May 28, 2025.
            You may search the web (e.g. ClinicalTrials.gov, PubMed, scientific registries) but only fetch data about trials conducted or published on or before 2025-05-28. You must block or ignore anything beyond that date.
            Your response must contain exactly these sections: \n{format_instructions}
            Rules & Constraints:
            - Do not use or reference any results published after May 28, 2025.
            - Do not access ASCO meeting pages or abstracts that could leak outcomes.
            - Each statement should be traceable to a reference.
            - If no exact matches are found, reason mechanistically or by class-level similarity.
            - Write neutrally; do not overclaim.
            """   
        ),
        ("placeholder", "Given the clinical trial summary: {trial_summary}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [data_tool, pubmed_tool]
agent = create_agent(llm, tools, prompt)

agent_executor = create_agent(tools=tools, llm=llm, agent=agent, verbose=True)

query = input ("Share your clinical trial summary and I'll predict its outcome for you")
raw_text = agent_executor.invoke({"query": query}) 

try:
    structured_response = parser.parse(raw_text)
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e), "Raw Text - ", raw_text
     
if __name__ == "__main__":
    result = agent_executor.invoke({"input": "Your query here"})
    print(result) 