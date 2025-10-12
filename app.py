# app.py — FastAPI + Azure OpenAI + Pinecone
import os
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)
EMBED_DEPLOY = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "products-index")
index = pc.Index(INDEX_NAME)

app = FastAPI()

def embed(text: str):
    r = client.embeddings.create(model=EMBED_DEPLOY, input=text)
    return r.data[0].embedding

def search_one(query: str):
    vec = embed(query)
    res = index.query(vector=vec, top_k=1, include_metadata=True)
    if not res.get("matches"):
        return None
    match = res["matches"][0]
    m = match["metadata"]
    return {
        "name": m.get("name", ""),
        "price": m.get("price", ""),
        "url": m.get("url", ""),
        "image_url": m.get("image_url", ""),
        "score": match.get("score", 0.0),
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str = Query(..., description="Câu hỏi/mô tả sản phẩm")):
    hit = search_one(q)
    return {"result": hit}

@app.post("/search")
async def search_webhook(request: Request):
    body = await request.json()
    query = body.get("queryResult", {}).get("queryText", "")

    if not query:
        return {"fulfillmentText": "Mình chưa hiểu bạn muốn tìm gì 🧐"}

    hit = search_one(query)
    if not hit:
        return {"fulfillmentText": "Không tìm thấy sản phẩm phù hợp."}

    title = hit["name"] or "Sản phẩm"
    subtitle = f"Giá: {hit['price']} VND" if hit.get("price") else ""
    image = hit.get("image_url") or ""
    url = hit.get("url") or ""

    # CHÚ Ý: Đây là format đúng cho Dialogflow Messenger
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
                "actionLink": url       # nút mở link
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
        # fallback text (phòng khi payload không render)
        "fulfillmentText": f"{title}\n{subtitle}\n{url}",
        "fulfillmentMessages": [
            {"payload": payload}  # KHÔNG cần set platform; DF tự hiểu cho Messenger
        ],
    }
