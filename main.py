"""
AI Product Classification Mini-Agent MVP - Main FastAPI Application
===================================================================

Multi-agent system for AI-powered product classification using:
- Semantic retrieval (FAISS + embeddings)
- LLM-based classification 
- Validation and feedback loops

Architecture:
User → API → IngestionAgent → RetrievalAgent → ClassifierAgent → ValidatorAgent → FeedbackAgent
"""

import yaml
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Import our agents
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent  
from agents.classifier_agent import ClassifierAgent
from agents.validator_agent import ValidatorAgent
from agents.feedback_agent import FeedbackAgent
from utils.vector_store import VectorStore


# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=config["api"]["title"],
    description="AI-powered product classification system using multi-agent architecture",
    version="1.0.0"
)

# Global agents (initialized on startup)
vector_store = None
agents = {}


# Pydantic models
class ProductRequest(BaseModel):
    description: str = Field(..., description="Product description to classify", min_length=10)
    include_context: Optional[bool] = Field(True, description="Include retrieval context in response")
    
class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from classification response")
    rating: int = Field(..., description="User rating (1-5)", ge=1, le=5)
    comments: Optional[str] = Field("", description="Optional user comments")

class ClassificationResponse(BaseModel):
    prediction: Dict[str, Any]
    validation: Dict[str, Any] 
    feedback: Dict[str, Any]
    retrieval_context: Optional[List[Dict[str, Any]]] = None
    session_id: str


@app.on_event("startup")
async def startup_event():
    """Initialize agents and vector store on startup."""
    global vector_store, agents
    
    print("Initializing AI Product Classification MVP...")
    
    try:
        # Check if required files exist
        csv_path = "data/catalog_sample.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Catalog file not found: {csv_path}")
        
        # Initialize vector store
        print("Loading vector store...")
        vector_store = VectorStore(csv_path, config["vector_db"]["path"])
        
        # Initialize agents
        print("Initializing agents...")
        agents = {
            "ingestion": IngestionAgent(csv_path),
            "retrieval": RetrievalAgent(vector_store, config["retrieval"]["top_k"]),
            "classifier": ClassifierAgent(config["llm_model"]),
            "validator": ValidatorAgent(csv_path),
            "feedback": FeedbackAgent(config["logging"]["file"])
        }
        
        print("AI Product Classification MVP startup complete!")
        
    except Exception as e:
        print(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Product Classification Mini-Agent MVP",
        "version": "1.0.0",
        "description": "AI-powered product classification using multi-agent architecture",
        "endpoints": {
            "classify": "/classify - Classify a product description",
            "feedback": "/feedback - Submit user feedback", 
            "health": "/health - System health check",
            "stats": "/stats - System statistics"
        }
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(request: ProductRequest):
    """
    Classify a product description using the multi-agent pipeline.
    
    Process:
    1. Load catalog data (IngestionAgent)
    2. Find similar products (RetrievalAgent)  
    3. Classify using LLM (ClassifierAgent)
    4. Validate prediction (ValidatorAgent)
    5. Log results (FeedbackAgent)
    """
    try:
        print(f"\nProcessing classification request...")
        print(f"Product: {request.description[:100]}...")
        
        # Step 1: Ingestion (verify catalog loaded)
        catalog_df = agents["ingestion"].run()
        
        # Step 2: Retrieval - find similar products
        similar_products = agents["retrieval"].run(request.description)
        context_summary = agents["retrieval"].get_context_summary(similar_products)
        
        # Step 3: Classification - predict codes using LLM
        prediction_json = agents["classifier"].run(request.description, context_summary)
        
        # Parse and validate the JSON response
        try:
            prediction = json.loads(prediction_json)
        except json.JSONDecodeError:
            # Try to extract JSON from response if LLM added extra text
            import re
            json_match = re.search(r'\{.*\}', prediction_json, re.DOTALL)
            if json_match:
                prediction = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse LLM response as JSON")
        
        # Step 4: Validation - check against catalog constraints  
        validation_result = agents["validator"].run(json.dumps(prediction))
        
        # Step 5: Feedback - log the complete result
        complete_result = {
            "input": request.description,
            "retrieval_context": similar_products,
            "raw_prediction": prediction,
            "validation": validation_result
        }
        
        feedback_result = agents["feedback"].run(complete_result)
        
        # Prepare response
        response_data = {
            "prediction": prediction,
            "validation": validation_result,
            "feedback": feedback_result,
            "session_id": feedback_result.get("session_id", "unknown")
        }
        
        # Include retrieval context if requested
        if request.include_context:
            response_data["retrieval_context"] = similar_products
        
        print(f"Classification complete for session {response_data['session_id']}")
        
        return response_data
        
    except Exception as e:
        print(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a classification session."""
    try:
        result = agents["feedback"].log_user_feedback(
            request.session_id,
            request.rating, 
            request.comments
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/health")
async def health_check():
    """System health check."""
    try:
        # Check vector store
        stats = vector_store.get_stats() if vector_store else {}
        
        # Check agents
        agent_status = {name: "ready" for name in agents.keys()} if agents else {}
        
        return {
            "status": "healthy",
            "vector_store": stats,
            "agents": agent_status,
            "config": {
                "model": config["llm_model"],
                "top_k": config["retrieval"]["top_k"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics and performance metrics."""
    try:
        stats = {
            "vector_store": vector_store.get_stats() if vector_store else {},
            "catalog_info": agents["ingestion"].get_schema() if "ingestion" in agents else {},
            "validation_constraints": agents["validator"].get_validation_stats() if "validator" in agents else {},
            "logs_summary": agents["feedback"].get_logs_summary() if "feedback" in agents else {}
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.get("/catalog/codes")
async def get_valid_codes():
    """Get all valid classification codes from the catalog."""
    try:
        if vector_store:
            return vector_store.get_valid_codes()
        else:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code retrieval failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        log_level=config["logging"]["level"].lower()
    )