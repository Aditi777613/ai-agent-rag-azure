"""
RAG Module — Document loading, chunking, and FAISS vector store management.

Supports Google Gemini embeddings (free tier) and OpenAI embeddings.
"""

import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
DOCUMENTS_DIR = "documents"


def _get_embeddings():
    """Return the appropriate embeddings model based on environment config."""
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    else:
        # Default: Google Gemini (free)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )


def build_vectorstore():
    """
    Load all documents from the documents/ folder, split into chunks,
    generate embeddings, and save the FAISS index locally.
    """
    from langchain_community.vectorstores import FAISS

    docs = []
    for path in glob.glob(os.path.join(DOCUMENTS_DIR, "*")):
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No documents found in '{DOCUMENTS_DIR}/' folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[SUCCESS] Split {len(docs)} documents into {len(chunks)} chunks.")

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[SUCCESS] FAISS index saved to '{FAISS_INDEX_PATH}/'")
    return vectorstore


def load_vectorstore():
    """Load a previously built FAISS index from disk."""
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(FAISS_INDEX_PATH):
        print("⚠️  FAISS index not found. Building from documents...")
        return build_vectorstore()

    embeddings = _get_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    build_vectorstore()
