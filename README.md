# NGS360 Chat Service

Standalone conversational AI service for NGS360, split out of the main REST API
(`APIServer`) so the LangChain/LangGraph dependency tree and the long-lived SSE
streaming workload don't live in — or slow down — the REST API.

## What it is

A thin FastAPI app that:

- Exposes `POST /api/v1/chat`, streaming replies in the **Vercel AI SDK UI
  Message Stream Protocol** (SSE), consumed by the React frontend's `useChat`.
- Runs a **LangGraph agent** (`create_agent`) backed by Claude on **Amazon
  Bedrock** (Converse API via `langchain-aws` `ChatBedrockConverse`).
- Verifies the **same JWT** the REST API issues (shared `JWT_SECRET_KEY`) — it
  does **not** touch the database.
- Gets project/run/sample data by **calling the REST API over HTTP** with the
  caller's bearer token, so the REST API stays the single source of truth.

```
React frontend ──POST /api/v1/chat──▶ ngs360-chat-service (this repo)
                                          │ agent tools call back over HTTP
                                          ▼
                                     NGS360 REST API (APIServer)
```

## Boundary with the REST API

| Concern        | Here                                            |
|----------------|-------------------------------------------------|
| LLM / agent    | ✅ owns it (LangChain, LangGraph, ChatBedrockConverse) |
| Auth           | verifies the shared JWT only (no DB, no API-key path) |
| Data           | ❌ no DB — calls `NGS360_API_URL` over HTTP      |
| Deploy/runtime | own image + process, streaming-tuned timeouts    |

## Routing

The frontend keeps calling `/api/v1/chat`. Point that path at this service via
an ALB path rule (`/api/v1/chat*` → this service's target group); everything
else continues to the REST API.

## Local dev

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in LLM_MODEL, AWS_REGION, JWT_SECRET_KEY, NGS360_API_URL
aws sso login          # any AWS identity with Bedrock invoke access
uvicorn main:app --reload --port 3100

# exercise it without the frontend:
python scripts/test_chat.py "Take me to run R-2024-01"
```

## Configuration (`.env`)

| Var | Purpose |
|-----|---------|
| `LLM_MODEL` | Bedrock model / inference-profile id (default `us.anthropic.claude-sonnet-4-6`) |
| `AWS_REGION` | Bedrock runtime region (default `us-east-1`) |
| `JWT_SECRET_KEY` / `JWT_ALGORITHM` | Must match the REST API so tokens verify |
| `NGS360_API_URL` | REST API base for tool lookups, e.g. `http://localhost:3000/api/v1` |
| `CLIENT_ORIGIN` | CORS allowed origin (the frontend) |

AWS access is **not** configured here: the Bedrock client uses the default
boto3 chain (the ECS task role in deployed environments). The role needs
`bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the model /
inference profile (and the underlying foundation models for cross-region
`us.` inference profiles).
