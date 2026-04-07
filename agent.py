"""
Agent Module — LLM agent with tool calling and session-based memory.

The agent decides whether to answer directly or search internal documents
using the RAG vector store. Supports Google Gemini, OpenAI, and Azure OpenAI.
"""

import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from rag import load_vectorstore

load_dotenv()

# ---------------------------------------------------------------------------
# Vector store (loaded once at module level)
# ---------------------------------------------------------------------------
vectorstore = load_vectorstore()

# ---------------------------------------------------------------------------
# Session memory store  (in-memory dict — resets on restart)
# ---------------------------------------------------------------------------
_memory_store: dict[str, ConversationBufferMemory] = {}


def get_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create conversation memory for a session."""
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    return _memory_store[session_id]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def search_documents(query: str) -> str:
    """Search internal company documents for information relevant to the query."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."

    sources = list({doc.metadata.get("source", "unknown") for doc in results})
    content = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )
    return f"SOURCES: {sources}\n\nCONTENT:\n{content}"


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _get_llm():
    """Return the appropriate chat model based on environment config."""
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0,
        )
    else:
        # Default: Google Gemini (free)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            convert_system_message_to_human=True,
        )


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful AI assistant for the company.
You have access to internal company documents including HR policies,
IT security policies, product FAQs, and expense/travel policies.

INSTRUCTIONS:
- Use the search_documents tool when the user asks about company policies,
  procedures, product information, HR questions, IT rules, or expense guidelines.
- Answer directly (without using tools) for general knowledge questions,
  greetings, math, or topics clearly unrelated to company documents.
- When using information from documents, ALWAYS cite the source file(s).
- Be concise but thorough in your answers.
- If the documents don't contain relevant information, say so honestly."""

tools = [
    Tool(
        name="search_documents",
        func=search_documents,
        description=(
            "Search the company's internal documents (HR policy, IT security "
            "policy, product FAQ, expense policy) for information relevant to the "
            "user's question. Use this for any question about company policies, "
            "procedures, products, or guidelines."
        ),
    )
]


def build_agent(session_id: str) -> AgentExecutor:
    """Build a LangChain agent with tools and session memory."""
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    memory = get_memory(session_id)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
