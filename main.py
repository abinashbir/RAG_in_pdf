from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from rag_pipeline import get_answer
import os

app = FastAPI()
APP_API_KEY = os.getenv("APP_API_KEY")  # separate from GEMINI_API_KEY

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query, x_api_key: str = Header(...)):
    if not APP_API_KEY or x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"answer": get_answer(q.question)}

@app.get("/")
def health():
    return {"status": "running"}