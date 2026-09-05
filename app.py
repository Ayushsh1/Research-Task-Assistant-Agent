import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent import build_agent_executor

load_dotenv()

st.set_page_config(page_title="Research & Task Agent", page_icon="🤖")
st.title("🤖 Research & Task Agent")


# --- API key handling ---------------------------------------------------
# The key is read only from the .env file — never shown or entered in the UI.
api_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("---")
    st.markdown(
        "**Tools available:**\n"
        "- 🔎 Web search\n"
        "- 📚 Wikipedia\n"
        "- 🧮 Calculator\n"
        "- 📝 Save / read notes"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env file and restart the app.")
    st.stop()

# --- Session state --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of HumanMessage / AIMessage

if "executor" not in st.session_state or st.session_state.get("_model") != model:
    st.session_state.executor = build_agent_executor(api_key, model=model)
    st.session_state._model = model

# --- Render chat history ---------------------------------------------------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Chat input --------------------------------------------------------
user_input = st.chat_input("Ask me anything, or tell me to save a note...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.executor.invoke(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                answer = result["output"]
            except Exception as e:
                answer = f"Something went wrong: {e}"
            st.markdown(answer)

    st.session_state.messages.append(AIMessage(content=answer))