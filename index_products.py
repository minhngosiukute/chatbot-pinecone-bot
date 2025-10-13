# index_products.py  (Azure OpenAI + pinecone-client 3.x)
import os, csv, time, sys
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

# ====== CẤU HÌNH ======
INDEX_NAME   = "products-index"
PC_REGION    = "us-east-1"
EMBED_DIM    = 1536      # text-embedding-3-small -> 1536; 3-large -> 3072
BATCH_SIZE   = 100
# ======================

load_dotenv()

# Azure OpenAI
AZURE_OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_EMBED_DEPLOY      = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

# Pinecone
PINECONE_API_KEY        = os.getenv("PINECONE_API_KEY")

if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, PINECONE_API_KEY]):
    raise RuntimeError(
        "Thiếu biến môi trường. Cần AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_API_VERSION, AZURE_EMBED_DEPLOY, PINECONE_API_KEY trong file .env"
    )

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VER,
)
pc = Pinecone(api_key=PINECONE_API_KEY)

# tạo index nếu chưa có
existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"⏳ Chưa có index '{INDEX_NAME}', đang tạo...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PC_REGION),
    )
    time.sleep(15)

index = pc.Index(INDEX_NAME)

def embed(text: str):
    r = client.embeddings.create(model=AZURE_EMBED_DEPLOY, input=text)
    return r.data[0].embedding

csv_path = os.path.join(os.path.dirname(__file__), "products.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Không thấy products.csv: {csv_path}")

batch, total_rows = [], 0

# MỌI THAO TÁC VỚI reader đều ở trong with
with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    # kiểm tra header
    headers = [c.strip() for c in (reader.fieldnames or [])]
    required = {"id", "name", "category", "description", "price", "local_url", "image_url", "category_id", "tags"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"CSV thiếu cột: {', '.join(missing)}")

    # duyệt từng dòng
    for row in reader:
        total_rows += 1
        doc = f"{row['name']} | {row['category']} | {row.get('tags','')} | {row['description']}"
        batch.append({
            "id": str(row["id"]),
            "values": embed(doc),
            "metadata": {
                "name": row["name"],
                "category": row["category"],
                "category_id": row["category_id"],
                "description": row["description"],
                "url": row["local_url"],
                "image_url": row["image_url"],
                 "price": float(row["price"]),
                "tags": row["tags"],
            }
        })

if not batch:
    print("⚠️ CSV trống.")
    sys.exit(0)

print(f"🚀 Bắt đầu nạp {len(batch)} sản phẩm vào Pinecone...")
for i in range(0, len(batch), BATCH_SIZE):
    index.upsert(batch[i:i+BATCH_SIZE])
    print(f"✅ Upsert {min(i+BATCH_SIZE, len(batch))}/{len(batch)}")

print(f"🎉 Hoàn tất nạp {total_rows} dòng dữ liệu vào Pinecone!")
