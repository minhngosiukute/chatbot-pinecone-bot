# app.py — FastAPI + Azure OpenAI + Pinecone (trả 1 sp tốt nhất)
import os
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

# ===== Azure OpenAI =====
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)
EMBED_DEPLOY = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

# ===== Pinecone =====
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
    if not res.get("matches"):
        return None
    match = res["matches"][0]
    m = match.get("metadata", {}) or {}
    return {
        "name": m.get("name"),
        "price": m.get("price"),
        "url": m.get("url"),
        "image_url": m.get("image_url"),
        "score": match.get("score"),
    }

def format_hit(hit: dict) -> str:
    name = hit.get("name", "Sản phẩm")
    price = hit.get("price", "—")
    url = hit.get("url", "")
    msg = f"Sản phẩm phù hợp nhất: {name}\nGiá: {price} VND"
    if url:
        msg += f"\nLink: {url}"
    return msg

@app.get("/health")
def health():
    return {"ok": True}

# GET /search?q=... — test nhanh trên trình duyệt
@app.get("/search")
def search_get(q: str = Query(..., description="Câu hỏi/mô tả sản phẩm")):
    hit = search_one(q)
    return {"result": hit}

# app.py (chỉ phần POST /search)
from fastapi import Request

@app.post("/search")
async def search_webhook(request: Request):
    body = await request.json()
    query = body.get("queryResult", {}).get("queryText", "")

    if not query:
        return {"fulfillmentText": "Tôi không hiểu bạn muốn tìm gì 🧐"}

    hit = search_one(query)
    if not hit:
        return {"fulfillmentText": "Xin lỗi, mình chưa tìm thấy sản phẩm phù hợp."}

    title = hit["name"] or "Sản phẩm"
    subtitle = f"Giá: {int(hit['price']):,} VND".replace(",", ".") if hit.get("price") else ""
    image = hit.get("image_url") or ""
    url = hit.get("url") or "#"

    # Trả về card để mọi kênh (kể cả Web Demo) render có ảnh + button
    return {
        "fulfillmentMessages": [
            {
                "card": {
                    "title": title,
                    "subtitle": subtitle,
                    "imageUri": image,
                    "buttons": [
                        {"text": "XEM CHI TIẾT", "postback": url}
                    ]
                }
            }
        ]
    }
