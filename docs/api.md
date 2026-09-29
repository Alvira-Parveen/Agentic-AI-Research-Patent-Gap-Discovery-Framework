# FastAPI Backend Documentation

## 1. Starting the Server
```bash
uvicorn src.main:app --reload --port 8000
```
Interactive Swagger API documentation is available at:
`http://localhost:8000/docs`

---

## 2. Endpoints Reference

### `GET /health`
Returns service status and active backend configurations.
**Response**:
```json
{
  "status": "healthy",
  "service": "PBL-3 Patent & Research Gap Discovery",
  "llm_provider": "mock",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "vector_store": "faiss"
}
```

### `GET /config`
Returns public configuration parameters without leaking API keys.

### `POST /analyze`
Analyzes an invention idea using a selected mode.
**Request Body**:
```json
{
  "idea": "An AI system that uses drone-based multispectral images to detect crop diseases and recommend variable-rate chemical treatment.",
  "mode": "multi_agent_rag",
  "domain": "Agriculture & Computer Vision"
}
```
**Response**: Returns full `FinalAnalysisReport` (Pydantic schema).

### `POST /compare`
Runs the idea through all three modes (Single LLM, LLM + RAG, Multi-Agent RAG) and returns side-by-side metrics and comparative outputs.

### `POST /search/patents` & `POST /search/papers`
Direct semantic vector retrieval against the patent or research paper corpus.
**Request Body**:
```json
{
  "query": "drone multispectral disease detection",
  "top_k": 3
}
```

### `POST /evaluate`
Triggers the comparative benchmark over test cases and returns summary metrics.
