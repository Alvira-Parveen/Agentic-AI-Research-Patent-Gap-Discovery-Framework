import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data


def test_api_config():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "embedding_model" in data
    assert "novelty_weights" in data
    # Verify no secret keys are exposed
    assert "openai_api_key" not in data


def test_api_analyze_single_llm():
    payload = {
        "idea": "An AI system that uses drone multispectral images to detect crop diseases and prescribe treatment.",
        "mode": "single_llm"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "single_llm"
    assert "final_summary" in data
    assert data["novelty"]["novelty_score"] is None


def test_api_analyze_multi_agent():
    payload = {
        "idea": "An AI system that uses drone multispectral images to detect crop diseases and prescribe treatment.",
        "mode": "multi_agent_rag"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "multi_agent_rag"
    assert data["novelty"]["is_heuristic"] is True
    assert "Prototype Multi-Factor Prior-Art Overlap Heuristic" in data["novelty"]["formula_explanation"]


def test_api_compare():
    payload = {
        "idea": "An AI system that uses drone multispectral images to detect crop diseases and prescribe treatment."
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "single_llm" in data
    assert "rag" in data
    assert "multi_agent_rag" in data
    assert data["single_llm"]["novelty"]["novelty_score"] is None
