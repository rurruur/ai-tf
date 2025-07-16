from fastapi import APIRouter, Depends
from dotenv import dotenv_values
import json

from app.services.langchain import LangchainService
from app.models import QueryRequest, QueryResponse

config = dotenv_values()

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)


def get_langchain_service() -> LangchainService:
    return LangchainService(config)

@router.get("/health")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "ok"}

@router.post("/search", response_model=QueryResponse)
def search_products(
    request: QueryRequest,
    langchain_service: LangchainService = Depends(get_langchain_service)
):
    """
    Search products using LangGraph-powered RAG system.
    """
    json_response = langchain_service.call(request.query)
    
    print(json.dumps(json_response, indent=2, ensure_ascii=False))
    
    return QueryResponse(**json_response)