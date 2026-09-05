"""
Tools available to the agent.

Each tool is a plain Python function wrapped with LangChain's @tool
decorator. The docstring of each function is what the LLM reads to decide
when and how to call the tool — keep them clear and specific.
"""

import os
import ast
import operator

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

NOTES_FILE = "notes.txt"

# ---------------------------------------------------------------------------
# Web search (no API key required)
# ---------------------------------------------------------------------------
_search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts you don't
    already know. Use this for anything time-sensitive or specific
    (e.g. 'latest AI Engineer job openings in India', 'current USD to INR rate').
    """
    try:
        return _search.run(query)
    except Exception as e:
        return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# Wikipedia lookup
# ---------------------------------------------------------------------------
_wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)


@tool
def wikipedia_lookup(query: str) -> str:
    """Look up a factual, encyclopedic topic on Wikipedia — good for
    definitions, historical facts, biographies, or background on well-known
    concepts (e.g. 'transformer architecture', 'Reserve Bank of India').
    """
    try:
        return _wiki.run(query)
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


# ---------------------------------------------------------------------------
# Safe calculator (no eval() — parses a restricted AST instead)
# ---------------------------------------------------------------------------
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '23 * 47 + 100' or '(15/3)**2'.
    Only use this for arithmetic — supports +, -, *, /, **, %, and parentheses.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


# ---------------------------------------------------------------------------
# Note saving — demonstrates simple persistent state
# ---------------------------------------------------------------------------
@tool
def save_note(note: str) -> str:
    """Save a short note or reminder to persistent storage for later
    reference. Use this when the user asks you to remember or note something
    down.
    """
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(note.strip() + "\n")
    return "Note saved."


@tool
def read_notes(_: str = "") -> str:
    """Read back all notes that have been saved so far. Call this when the
    user asks what notes have been saved or to recall past notes.
    """
    if not os.path.exists(NOTES_FILE):
        return "No notes saved yet."
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else "No notes saved yet."


def get_tools():
    return [web_search, wikipedia_lookup, calculator, save_note, read_notes]
