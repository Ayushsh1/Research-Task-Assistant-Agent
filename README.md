# Research & Task Agent

A tool-using AI agent built with **LangChain** and **OpenAI**, wrapped in a **Streamlit** chat UI.

## What it does

The agent can autonomously decide when to:
- 🔎 **Search the web** (DuckDuckGo — no extra API key needed)
- 📚 **Look up Wikipedia** for factual/background info
- 🧮 **Calculate** math expressions safely (no `eval`)
- 📝 **Save and recall notes** (simple persistent state)

It uses LangChain's `create_tool_calling_agent`, which relies on OpenAI's native function-calling — the model itself decides which tool to call and with what arguments.

## Project structure

```
agent-project/
├── app.py          # Streamlit chat UI
├── agent.py        # Agent + LLM setup (AgentExecutor)
├── tools.py        # Tool definitions (search, wiki, calculator, notes)
├── requirements.txt
└── .env
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your OpenAI API key**

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

   This opens a chat interface in your browser (usually `http://localhost:8501`).

## Example prompts to try

- "What's the latest news on AI regulation in India?"
- "What is the transformer architecture in deep learning?"
- "Calculate (245 * 12) / 7"
- "Save a note: follow up with recruiter on Monday"
- "What notes have I saved?"

## How it works (for your resume / interview talking points)

- **Agent type**: OpenAI tool-calling agent (`create_tool_calling_agent`) — the LLM outputs structured tool calls rather than parsing free text, which is more reliable than older ReAct-style prompting.
- **Memory**: Conversation history is passed back into the prompt each turn via `MessagesPlaceholder`, so the agent has context across turns.
- **Safety**: The calculator tool uses Python's `ast` module to parse expressions instead of `eval()`, preventing arbitrary code execution.
- **Extensibility**: Adding a new capability is as simple as writing a new `@tool`-decorated function in `tools.py` and adding it to `get_tools()`.

## Extending this project

Ideas to build on this for your portfolio:
- Add a **retrieval tool** (RAG) over a PDF/resume/docs using a vector store (e.g. Chroma or FAISS).
- Swap `AgentExecutor` for **LangGraph** to add branching logic or a multi-agent handoff.
- Add a tool that calls a real job-search API (e.g. scraping or an API for LinkedIn/Naukri listings) — ties directly into your own job search.
- Deploy the Streamlit app to **Streamlit Community Cloud** or **Render** for a live demo link on your resume.
