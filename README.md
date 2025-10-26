# Clinical Trial Outcome Prediction Agent
AI agent that predicts whether clinical trial endpoints will be met using historical data, confidence flags, and structured outputs.
AI Agent was built using Langchain.

### Features
- Predict trial endpoints (Met / Unmet)
- Confidence flag system (Green/Yellow/Red)
- Tool-based evidence gathering (data_tool, pubmed_tool)
- Automatic saving of JSON outputs

### Installation
<img width="524" height="130" alt="image" src="https://github.com/user-attachments/assets/56036c79-2ece-4946-a27c-403ee73b8cab" />

### Usage
<img width="636" height="125" alt="image" src="https://github.com/user-attachments/assets/4e13030d-3777-48b5-8fbe-d2e8a0a152e0" />

### Example Output
<img width="1919" height="573" alt="image" src="https://github.com/user-attachments/assets/fb726939-1b8d-43da-a2ad-2877e8f01588" />

### Tools 
Built the following tools from scratch using Python: 
- data_tool for searching through historical data from already prepared dataset
- pubmed_tool for retrieving previously published clinical trial data to help agent's reasoning process
- save_tool to automatically save outputs into a text file

### Agent Workflow
LLM (Gemini 2.5 Pro) -> Tools -> JSON Parser -> Prediction Output

### Agent's Potential Impact
In a $50B+ clinical research analytics market, this AI agent has the potential to: 
- Reduce R&D costs by identifying low-probability trials before they begin.
- Accelerate decision-making, helping teams focus on the most promising therapies.
- Support regulatory and portfolio strategy with transparent, evidence-based insights.
- Empower researchers with accessible, AI-assisted analysis tools that enhance, rather than replace, human expertise.

### Watch the demo video
Link: https://drive.google.com/file/d/1VO51DCniS2L5R8cnUrxQ6f670gruQSSe/view?usp=drivesdk

### Note:
I have made this repo into a template for easy replication. This is a simple viable model. Feel free to clone and build upon it.
