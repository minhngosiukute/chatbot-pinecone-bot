# app.py — FastAPI + Azure OpenAI + Pinecone (trả 1 sp tốt nhất)
import os
from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

# Azure OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)
EMBED_DEPLOY = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "products-index")
index = pc.Index(INDEX_NAME)

app = FastAPI(title="Product Search API")

def embed(text: str):
    r = client.embeddings.create(model=EMBED_DEPLOY, input=text)
    return r.data[0].embedding

class SearchRequest(BaseModel):
    query: str

def search_one(query: str):
    vec = embed(query)
    res = index.query(vector=vec, top_k=1, include_metadata=True)
    if not res["matches"]:
        return None
    m = res["matches"][0]["metadata"]
    return {
        "name": m.get("name"),
        "price": m.get("price"),
        "url": m.get("url"),
        "image_url": m.get("image_url"),
        "score": res["matches"][0]["score"],
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str = Query(..., description="Câu hỏi/mô tả sản phẩm")):
    hit = search_one(q)
    return {"result": hit}

@app.post("/search")
def search_body(req: SearchRequest):
    hit = search_one(req.query)
    return {"result": hit}
