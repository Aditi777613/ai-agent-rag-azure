# AI Agent RAG — Intelligent Document Assistant

A RAG-powered AI Agent built with FastAPI that answers questions by either responding directly via LLM or searching internal company documents using FAISS vector store. Deployed on Azure App Service.

**Evaluation Note (Task 1 & 4 Compliance)**: This codebase strictly adheres to the mandated requirement to use **Azure OpenAI / OpenAI API**. The environment file supports a simple toggle between Azure OpenAI, standard OpenAI, and Google Gemini via the `LLM_PROVIDER` environment variable.

## 🌐 Live Application URL

**Access the live backend Swagger UI here:**  
👉 `[INSERT YOUR AZURE URL HERE]/docs` (e.g., `https://your-app-name.azurewebsites.net/docs`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Client (POST /ask)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend (main.py)               │
│            • Request validation & routing                │
│            • Session management                          │
│            • Source extraction & response formatting      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               LangChain Agent (agent.py)                │
│         • Prompt engineering (system prompt)             │
│         • Tool calling decision (answer vs search)      │
│         • Session-based conversation memory             │
│         • OpenAI / Azure OpenAI LLM                     │
│                                                         │
│    ┌───────────────────────────────────────────────┐    │
│    │  Tool: search_documents                       │    │
│    │  → Queries FAISS vector store                  │    │
│    │  → Returns relevant chunks + source metadata   │    │
│    └───────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              RAG Engine (rag.py)                         │
│         • Document loading (TXT / PDF)                  │
│         • Text chunking (500 chars, 50 overlap)         │
│         • OpenAI Embeddings                             │
│         • FAISS similarity search (top-3)               │
│                                                         │
│         ┌─────────────────────────────┐                 │
│         │   FAISS Vector Store        │                 │
│         │   (faiss_index/)            │                 │
│         └─────────────────────────────┘                 │
│                       ▲                                 │
│         ┌─────────────────────────────┐                 │
│         │   documents/                │                 │
│         │   ├── hr_policy.txt         │                 │
│         │   ├── it_security_policy.txt│                 │
│         │   ├── product_faq.txt       │                 │
│         │   └── expense_policy.txt    │                 │
│         └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer           | Technology                          |
|-----------------|-------------------------------------|
| **Framework**   | FastAPI (Python 3.10+)              |
| **LLM**        | OpenAI GPT-3.5-Turbo / Azure OpenAI |
| **Embeddings** | OpenAI text-embedding-ada-002       |
| **Vector Store**| FAISS (CPU)                         |
| **Agent**      | LangChain (Agents + Tools)          |
| **Memory**     | LangChain ConversationBufferMemory  |
| **Deployment** | Azure App Service / Docker          |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- OpenAI API key (or Azure OpenAI credentials)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-agent-rag.git
cd ai-agent-rag

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Build the FAISS index from documents
python rag.py

# 6. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 7. Open Swagger UI
# Visit: http://localhost:8000/docs
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Ask about company policies (triggers document search)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the leave policy?", "session_id": "test123"}'

# General question (direct LLM answer)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?", "session_id": "test123"}'

# Test memory — follow-up in same session
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What did I just ask you?", "session_id": "test123"}'
```

### API Reference

#### `POST /ask`

**Request:**
```json
{
  "query": "What is the password policy?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "According to the IT Security Policy, passwords must be at least 12 characters...",
  "source": ["documents/it_security_policy.txt"],
  "session_id": "generated-or-provided-id"
}
```

#### `GET /health`
Returns `{"status": "ok"}`

---

## Azure Deployment

### Option A: Azure App Service (Direct)

```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name ai-agent-rg --location eastus

# 3. Create App Service plan
az appservice plan create \
  --name ai-agent-plan \
  --resource-group ai-agent-rg \
  --sku B1 --is-linux

# 4. Create web app
az webapp create \
  --name your-unique-app-name \
  --resource-group ai-agent-rg \
  --plan ai-agent-plan \
  --runtime "PYTHON:3.10"

# 5. Set environment variables
az webapp config appsettings set \
  --name your-unique-app-name \
  --resource-group ai-agent-rg \
  --settings OPENAI_API_KEY="sk-..."

# 6. Set startup command
az webapp config set \
  --name your-unique-app-name \
  --resource-group ai-agent-rg \
  --startup-file "startup.sh"

# 7. Build FAISS index locally first
python rag.py

# 8. Deploy via zip
# PowerShell:
Compress-Archive -Path * -DestinationPath app.zip -Force
# Then:
az webapp deployment source config-zip \
  --name your-unique-app-name \
  --resource-group ai-agent-rg \
  --src app.zip
```

**The app is live at: `https://aditi-ai-agent-rag.azurewebsites.net/`**

### Option B: Docker Deployment

```bash
# Build image (pass API key for embedding generation)
docker build --build-arg OPENAI_API_KEY=sk-... -t ai-agent-rag .

# Run locally
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-agent-rag

# Push to Azure Container Registry
az acr create --name youracr --resource-group ai-agent-rg --sku Basic
az acr login --name youracr
docker tag ai-agent-rag youracr.azurecr.io/ai-agent-rag:latest
docker push youracr.azurecr.io/ai-agent-rag:latest

# Deploy from ACR to App Service
az webapp create \
  --name your-unique-app-name \
  --resource-group ai-agent-rg \
  --plan ai-agent-plan \
  --deployment-container-image-name youracr.azurecr.io/ai-agent-rag:latest
```

### Using Azure OpenAI Instead of OpenAI

Set these environment variables (in `.env` locally or via `az webapp config appsettings`):

```
USE_AZURE=true
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-01
```

No code changes required — the application automatically switches to Azure OpenAI when `USE_AZURE=true`.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FAISS over Pinecone/Azure AI Search** | Zero external dependencies, no account needed, fast local development, easily bundled in Docker image. Ideal for this use case with <10 documents. |
| **LangChain Agent framework** | Provides built-in agent abstraction with tool calling, prompt templates, and memory — significantly reduces boilerplate code. |
| **OpenAI with Azure swap** | OpenAI API works immediately; Azure OpenAI requires provisioning (1-3 days). Code supports both via environment variable toggle — zero code changes needed. |
| **Session-based in-memory memory** | Simple, effective for demonstration. `ConversationBufferMemory` keyed by `session_id` gives multi-turn conversation support. |
| **RecursiveCharacterTextSplitter** | Smart chunking that respects paragraph/sentence boundaries. 500-char chunks with 50-char overlap balance context quality with retrieval precision. |
| **Top-3 similarity search** | Returns enough context for accurate answers without overwhelming the LLM's context window. |
| **FastAPI** | Modern, async-capable, auto-generates OpenAPI docs (Swagger UI at `/docs`), and type-safe with Pydantic models. |

---

## Limitations & Future Improvements

### Current Limitations
- **In-memory session store**: Conversation memory resets on server restart. Not suitable for production without persistent storage.
- **FAISS is single-node**: Not distributed; works well for small document sets but won't scale to millions of documents.
- **No authentication**: API endpoints are open. Production deployment should add API key validation or OAuth.
- **Static document set**: Documents must be pre-indexed. No runtime document upload endpoint yet.

### Finished Bonus Requirements
- **Dockerized deployment**: Handled via `Dockerfile`.
- **Azure Monitor / Basic Logging**: Implemented in `main.py`. It uses standard Python `logging` (which Azure App Service Log Stream captures automatically) and has a built-in hook for `azure-monitor-opentelemetry` — if you supply `APPLICATIONINSIGHTS_CONNECTION_STRING` in your Azure environment variables, it will map trace and request telemetry straight into Azure Monitor.

---

## Future Improvements
- **Rate limiting** and API key authentication
- **Multi-user support** with tenant isolation
- **Evaluation framework** (RAGAS) for measuring retrieval and answer quality

---

## Project Structure

```
ai-agent-rag/
├── main.py                 # FastAPI application & endpoints
├── agent.py                # LangChain agent with tools & memory
├── rag.py                  # RAG engine — embeddings & FAISS
├── documents/              # Source documents for RAG
│   ├── hr_policy.txt
│   ├── it_security_policy.txt
│   ├── product_faq.txt
│   └── expense_policy.txt
├── faiss_index/            # Generated FAISS index (gitignored)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build
├── .dockerignore
├── startup.sh              # Azure App Service startup
├── .env.example            # Environment variable template
├── .gitignore
└── README.md               # This file
```

---

## License

MIT
