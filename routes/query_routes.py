from fastapi import APIRouter, HTTPException
from models.query_models import QueryRequest, QueryResponse
from services.conversational_service import process_query

query_router = APIRouter()

@query_router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        answer = process_query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
