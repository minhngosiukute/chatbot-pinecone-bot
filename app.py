# app.py — FastAPI + Azure OpenAI + Pinecone (local-only tweaks)
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, Request
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

# ---------- Azure OpenAI ----------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)
EMBED_DEPLOY = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

# ---------- Pinecone ----------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "products-index")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "products")  # có/không đều chạy
index = pc.Index(INDEX_NAME)

app = FastAPI()

# ---------- Helpers ----------
def embed(text: str):
    r = client.embeddings.create(model=EMBED_DEPLOY, input=text)
    return r.data[0].embedding

def _pick_url(m: Dict[str, Any]) -> str:
    # hỗ trợ cả "url" và "local_url"
    return (m.get("url") or m.get("local_url") or "").strip()

def _normalize_hit(match: Dict[str, Any]) -> Dict[str, Any]:
    m = match.get("metadata", {}) or {}
    return {
        "id": m.get("id") or m.get("Id") or "",
        "name": m.get("name", ""),
        "category": m.get("category", ""),
        "tags": m.get("tags", ""),
        "price": m.get("price", ""),
        "url": _pick_url(m),
        "image_url": m.get("image_url", ""),
        "score": float(match.get("score", 0.0)),
    }

def search_one(query: str, top_k: int = 1) -> Optional[Dict[str, Any]]:
    vec = embed(query)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE if NAMESPACE else None
    )
    matches = res.get("matches") or []
    if not matches:
        return None
    # Lấy best hit; nếu muốn list nhiều, trả về matches ở /search (bên dưới có)
    return _normalize_hit(matches[0])

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(
    q: str = Query(..., description="Câu hỏi/mô tả sản phẩm"),
    top_k: int = Query(1, ge=1, le=10, description="Số kết quả muốn lấy (mặc định 1)")
):
    vec = embed(q)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE if NAMESPACE else None
    )
    matches = res.get("matches") or []
    results = [_normalize_hit(m) for m in matches]
    return {"results": results, "count": len(results)}

@app.post("/search")
async def search_webhook(request: Request):
    body = await request.json()
    query = body.get("queryResult", {}).get("queryText", "")

    if not query:
        return {"fulfillmentText": "Mình chưa hiểu bạn muốn tìm gì 🧐"}

    hit = search_one(query, top_k=1)
    if not hit:
        return {"fulfillmentText": "Không tìm thấy sản phẩm phù hợp."}

    title = hit["name"] or "Sản phẩm"
    subtitle = f"Giá: {hit['price']} VND" if hit.get("price") else ""
    image = hit.get("image_url") or ""
    url = hit.get("url") or ""

    payload = {
        "richContent": [[
            {
                "type": "image",
                "rawUrl": image,        # nên là https công khai để chắc chắn hiện
                "accessibilityText": title
            },
            {
                "type": "info",
                "title": title,
                "subtitle": subtitle,
                "actionLink": url
            },
            {
                "type": "button",
                "icon": {"type": "launch", "color": "#FFFFFF"},
                "text": "Xem chi tiết",
                "link": url
            }
        ]]
    }

    return {
        "fulfillmentText": f"{title}\n{subtitle}\n{url}",
        "fulfillmentMessages": [{"payload": payload}],
    }
