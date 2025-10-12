# query_products.py  (Azure OpenAI + Pinecone search - chỉ 1 sản phẩm tốt nhất)
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

# --- Nạp biến môi trường ---
load_dotenv()

AZURE_OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_EMBED_DEPLOY      = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")
PINECONE_API_KEY        = os.getenv("PINECONE_API_KEY")
INDEX_NAME              = "products-index"

# --- Khởi tạo client ---
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VER,
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Hàm tạo embedding ---
def embed(text: str):
    r = client.embeddings.create(model=AZURE_EMBED_DEPLOY, input=text)
    return r.data[0].embedding

# --- Hỏi người dùng ---
query = input("🔎 Nhập mô tả sản phẩm bạn cần: ").strip()
if not query:
    print("❌ Bạn chưa nhập gì cả.")
    exit()

# --- Tạo vector truy vấn ---
vec = embed(query)

# --- Tìm top 1 ---
res = index.query(
    vector=vec,
    top_k=1,
    include_metadata=True
)

# --- Hiển thị kết quả ---
if not res["matches"]:
    print("❌ Không tìm thấy sản phẩm phù hợp.")
else:
    match = res["matches"][0]
    m = match["metadata"]
    print("\n✅ Sản phẩm phù hợp nhất:")
    print(f"🏷️ Tên: {m.get('name', '(Không tên)')}")
    print(f"💰 Giá: {m.get('price', 'Chưa rõ')} VNĐ")
    print(f"🔗 Link: {m.get('url', '')}")
    if m.get("image_url"):
        print(f"🖼 Hình ảnh: {m['image_url']}")
    print(f"📊 Độ tương đồng: {match['score']:.3f}")
