from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from rag_pipeline import get_answer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
APP_API_KEY = os.getenv("APP_API_KEY")

if not APP_API_KEY:
    logger.warning("APP_API_KEY not set in environment variables")

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(q: Query, x_api_key: str = Header(...)):
    if not APP_API_KEY or x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        answer = get_answer(q.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def health():
    return {"status": "running"}