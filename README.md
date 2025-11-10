# AI Product Classification MVP

AI-Powered Product Classification System using a multi-agent architecture.  
This repository contains a production-ready demonstration of how to build intelligent product classification systems using semantic search, large language models (LLMs), FAISS, and multi-agent orchestration.

## Overview

This MVP demonstrates an enterprise-grade AI system that automatically classifies products into a catalog taxonomy by combining:
- Semantic retrieval using embeddings and FAISS
- LLM-based classification for prediction of taxonomy codes
- Rule-based validation to ensure business constraints
- Feedback and logging for continuous improvement

## Architecture

Simple pipeline:

```
Input Description  -> IngestionAgent -> RetrievalAgent -> ClassifierAgent -> ValidatorAgent -> FeedbackAgent
```

Each agent is specialized:
- IngestionAgent: loads and validates catalog and product data
- RetrievalAgent: performs semantic search (FAISS)
- ClassifierAgent: uses an LLM to predict classification codes
- ValidatorAgent: checks predictions against business constraints
- FeedbackAgent: logs results and collects user feedback

## Quick Start

Prerequisites:
- Python 3.8+
- OpenAI API key
- 8GB+ RAM (recommended for FAISS indexing)

Clone and install:

```bash
git clone https://github.com/nagken/mvpaiagent.git
cd mvpaiagent
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
```

Run demo:

```bash
python demo.py
```

Run API server (FastAPI):

```bash
python main.py
# then visit http://localhost:8000/docs
```

## Project structure

mvpaiagent/mvpaiagent/
- agents/
  - ingestion_agent.py
  - retrieval_agent.py
  - classifier_agent.py
  - validator_agent.py
  - feedback_agent.py
- utils/
  - vector_store.py
- data/
  - catalog_sample.csv
  - catalog_index.faiss (generated)
- logs/
  - agent_logs.jsonl (generated)
- config.yaml
- requirements.txt
- main.py
- orchestrator.py
- demo.py
- README.md

## Configuration

Edit `config.yaml` to customize behavior. Example config options:

```yaml
llm_model: "gpt-3.5-turbo"
vector_db:
  path: "data/catalog_index.faiss"
  dimension: 1536
retrieval:
  top_k: 3
  threshold: 0.8
```

You can also set environment variables:
```bash
export OPENAI_API_KEY="your_key_here"
export LOG_LEVEL="INFO"
export LLM_MODEL="gpt-4"  # optional
```

## API

Endpoints:
- POST /classify — Classify product description
- POST /feedback — Submit user feedback
- GET /health — System health check
- GET /stats — Performance statistics (optional)

Example request:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"description": "Dell Ultrasharp 27-inch 4K Monitor", "include_context": true}'
```

Example response:

```json
{
  "prediction": {
    "code_level_1": "MONITORS",
    "code_level_2": "DISPLAY",
    "vendor": "Dell",
    "price_range": "HIGH",
    "confidence": 0.92
  },
  "validation": {
    "is_valid": true,
    "validation_errors": []
  },
  "session_id": "session_20241110_143052_a1b2c3d4"
}
```

## Usage examples

Interactive demo:

```bash
python demo.py
```

Test a single product via orchestration script:

```bash
python orchestrator.py "Dell XPS 13 laptop with 16GB RAM"
```

Run the API server:

```bash
python main.py
# Visit http://localhost:8000/docs
```

## Docker

A minimal Dockerfile example:

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]
```

## Features

- Multi-agent architecture with modular agents
- Semantic search using FAISS and OpenAI embeddings
- LLM-based product categorization
- Rule-based validation to enforce business constraints
- Audit logging and feedback capture
- REST API for integration and deployment
- Interactive demo and sample data (20 products)

## Sample data

The repo includes ~20 sample products across categories such as:
- Laptops (Business, Consumer, Gaming)
- Monitors (Display)
- Accessories (Peripherals, Gaming)
- Networking (Routers, Wireless)
- Storage (SSDs, External drives)
- Processors, Graphics, etc.

## Testing & Support

- Quick tests:
  - Run `python demo.py`
  - Run `python orchestrator.py "Product description"`
  - Run `python main.py` and use `/docs`
- Logs: `logs/agent_logs.jsonl`
- Repo: https://github.com/nagken/mvpaiagent

For questions or issues, check logs and the `/health` endpoint. Update `config.yaml` for configuration changes.

## Business value

- Faster product classification compared to manual processes
- High accuracy augmented by validation layer
- Scalable to thousands of products per hour
- Consistent taxonomy mapping for onboarding and migrations

---

Built with dedication for AI innovation.