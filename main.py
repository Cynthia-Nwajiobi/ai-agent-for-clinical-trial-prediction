from dotenv import load_dotenv 
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import data_tool, pubmed_tool, save_tool, save_to_file
import json
import re

load_dotenv()

class ClinicalTrialResponse(BaseModel):
    predicted_endpoint: str
    justification: str
    flag: str
    rationale: str
    references: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
parser = PydanticOutputParser(pydantic_object=ClinicalTrialResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ( 
            "system", 
            """
            You are a clinical trial outcome prediction agent passionate about longevity and anti-aging trials and oncology research.
            Your task is to analyze a clinical trial and predict whether its **primary endpoint** will be **Met** or **Unmet**, based on historical data and mechanistic reasoning.

            EVIDENCE-BASED REASONING:
            - Use the `data_tool` first — this is your internal knowledge base containing historical trial data, patterns, and prior outcomes.
            - If the evidence from `data_tool` is insufficient, use the `pubmed_tool` to retrieve additional pre-2025 literature.
            - You are **strictly prohibited** from using or referencing:
                Any data, study, or publication **after May 28, 2025**.
                Any outcomes or analyses from the **2025 ASCO Meeting**.
                Ignore any such results completely.

            PREDICTION TASK:
            - Predict whether the trial’s primary endpoint will be **Met** or **Unmet** (it must be one of these two).
            - Base your reasoning on historical analogs, trial design, intervention mechanism, patient population, and prior success rates.
            - Summarize your reasoning clearly and factually, avoiding speculation.

            FLAG LOGIC:
            - Green Flag: High likelihood of success (>80%) — strong historical or mechanistic support.
            - Yellow Flag: Moderate likelihood of success (50–80%) — mixed evidence or uncertainty.
            - Red Flag: Low likelihood of success (<50%) — weak or unfavorable evidence.

            OUTPUT DETAILS:
            - The JSON MUST include a concise `rationale` string explaining WHY the flag was chosen (cite the key evidence and a numeric confidence percentage, e.g. "Confidence: 85%").
            - The JSON MUST include provide recommendations for future trial designs or considerations to improve success likelihood.
            - The `references` list MUST include all PubMed IDs or data sources you used.

            RESPONSE FORMAT:
            - Structure your response as per the provided format.
            - Your response must contain exactly these sections: \n{format_instructions}
            - Return VALID JSON Only - Ensure the JSON is complete and correctly formatted.
            - If you are unable to reach a conclusion, state so clearly in the justification.
            - Once the JSON output is generated, call the `save_tool` to save it as a text file.
            - Pass the entire JSON string as the `content` argument.

            IMPORTANT: Do NOT use markdown code fences (for example, ```json or ```). The final assistant message MUST be exactly the JSON object and nothing else. If you call tools, call them first, then output the JSON as the final assistant message.

            """   
        ),
        ("human", "Given the clinical trial summary: {trial_summary}\n\nQuery: {query}\n\n{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [data_tool, pubmed_tool, save_tool]
agent = create_tool_calling_agent(llm, tools, prompt)

# Turn off verbose tool traces to avoid mixing logs into the assistant output
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

if __name__ == "__main__":
    query = input("Enter your clinical trial summary: ")
    print("🔄 Analyzing trial with historical data and PubMed... Please wait...")
    result = agent_executor.invoke({"query": query, "trial_summary": query})

    # Try to extract agent output text
    raw = None
    if isinstance(result, dict):
        raw = result.get("output") or result.get("text") or result.get("result")
    elif isinstance(result, str):
        raw = result

    raw = (raw or "").strip()

    def try_extract_json(s: str):
        """Attempt to find and decode the first JSON object inside s."""
        # strip common markdown fences and leading text
        s_clean = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        # attempt to find the first JSON object or array
        decoder = json.JSONDecoder()
        for m in re.finditer(r"[\{\[]", s_clean):
            start = m.start()
            try:
                obj, end = decoder.raw_decode(s_clean[start:])
                return obj
            except Exception:
                continue
        return None

    if not raw:
        raw = str(result)

    # Try structured parsing
    try:
        structured_response = parser.parse(raw)
        print("\nStructured Response:")
        print(structured_response)
        # save parsed structure to file using the helper
        path = save_to_file(json.dumps(structured_response.model_dump(), indent=2))
        print("\nSaved structured response to file:", path)
    except Exception as e1:
        extracted = try_extract_json(raw)
        if extracted is not None:
            try:
                structured_response = ClinicalTrialResponse.parse_obj(extracted)
                print("\nStructured Response (extracted):")
                print(structured_response)
                path = save_to_file(json.dumps(structured_response.dict(), indent=2))
                print("\nSaved structured response to file:", path)
            except Exception as e2:
                print("Found JSON but failed to validate:", e2)
                path = save_to_file(raw)
                print("Saved raw output to:", path)
        else:
            path = save_to_file(raw)
            print("Error parsing response. Raw output saved to:", path)
            print("For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE")