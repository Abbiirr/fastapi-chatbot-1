from fastapi import FastAPI
from routes.query_routes import query_router

app = FastAPI()

# Include routes
app.include_router(query_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
