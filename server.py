#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from langserve import add_routes
from agents.assistant import assistant
from agents.data_pipeline import process_all_csv_files, search_vectors

# API Tags for Swagger organization
tags_metadata = [
    {
        "name": "Pipeline",
        "description": "Data pipeline operations for CSV processing and vector storage",
    },
    {
        "name": "Assistant",
        "description": "LangChain AI assistant endpoints",
    },
]

app = FastAPI(
    title="LangChain Agent API",
    version="1.0",
    description="""
## LangChain Code Agent API

This API provides:
- **Data Pipeline**: Process CSV files and store as vectors in Supabase
- **Vector Search**: Search the vector database for similar documents
- **AI Assistant**: Interactive AI assistant with various tools

### Getting Started
1. Upload CSV files to the `uploads/` folder
2. Call `/pipeline/start` to process and embed the data
3. Use `/pipeline/search` to query the vector database
4. Use `/assistant/playground/` for interactive AI chat
    """,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

add_routes(
    app,
    assistant,
    path="/assistant",
)


# Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text to find similar documents")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    filter_column: Optional[str] = Field(None, description="Column name to filter by (e.g., 'category')")
    filter_value: Optional[str] = Field(None, description="Value to filter on")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "countries in Europe",
                "top_k": 5,
                "filter_column": "continent",
                "filter_value": "Europe"
            }
        }


class PipelineResponse(BaseModel):
    status: str
    documents_processed: int


class SearchResponse(BaseModel):
    status: str
    results: Any


@app.get("/", tags=["General"])
async def root():
    """
    API root - returns available endpoints.
    """
    return {
        "message": "LangChain Agent API",
        "docs": "/docs",
        "endpoints": {
            "pipeline_start": "/pipeline/start",
            "pipeline_search": "/pipeline/search",
            "assistant_playground": "/assistant/playground/"
        }
    }


@app.post(
    "/pipeline/start",
    response_model=PipelineResponse,
    tags=["Pipeline"],
    summary="Start Data Pipeline",
    description="Process all CSV files in the uploads folder, embed them, and store in Supabase."
)
async def start_pipeline():
    try:
        count = process_all_csv_files()
        return {"status": "success", "documents_processed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/pipeline/search",
    response_model=SearchResponse,
    tags=["Pipeline"],
    summary="Search Vector Database",
    description="Search for similar documents using semantic similarity. Optionally filter by metadata columns."
)
async def search_pipeline(request: SearchRequest):
    try:
        # Build filter metadata dict from column/value params
        filter_metadata = None
        if request.filter_column and request.filter_value:
            filter_metadata = {request.filter_column: request.filter_value}
        
        results = search_vectors(
            query=request.query,
            filter_metadata=filter_metadata,
            top_k=request.top_k
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return RedirectResponse(url="/assistant/playground/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


