"""
Builds the LangChain agent: an OpenAI chat model + tool-calling agent
+ AgentExecutor, with conversation memory.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import get_tools

SYSTEM_PROMPT = """You are a helpful, precise research and task assistant.

You have access to tools for web search, Wikipedia lookups, calculations,
and saving/reading notes. Use a tool whenever it would make your answer
more accurate or current — don't guess at facts you can look up.

Be concise and direct in your final answers. If you used a tool, briefly
mention what you found rather than dumping raw tool output."""


def build_agent_executor(openai_api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3):
    """Returns an AgentExecutor ready to run with a chat_history + input."""
    if not openai_api_key:
        raise ValueError("OpenAI API key is required.")

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=openai_api_key,
    )

    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )
    return executor
