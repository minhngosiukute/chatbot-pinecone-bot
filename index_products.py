# index_products.py  (Azure OpenAI + pinecone-client 3.x, CSV -> Pinecone)
import os, csv, time, sys, io
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

# ====== CẤU HÌNH ======
INDEX_NAME   = os.getenv("PINECONE_INDEX", "products-index")
PC_REGION    = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_DIM    = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small=1536; 3-large=3072
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "100"))
NAMESPACE    = os.getenv("PINECONE_NAMESPACE", "products")  # có thể để trống
CSV_NAME     = os.getenv("PRODUCTS_CSV", "products.csv")    # đổi nếu khác
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
        "AZURE_OPENAI_API_VERSION, AZURE_EMBED_DEPLOY, PINECONE_API_KEY trong .env"
    )

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VER,
)
pc = Pinecone(api_key=PINECONE_API_KEY)

def _list_index_names() -> List[str]:
    # pinecone-client 3.x có thể trả về list dict hoặc object; xử lý cả hai
    names = []
    for i in pc.list_indexes():
        if isinstance(i, dict):
            names.append(i.get("name"))
        else:
            names.append(getattr(i, "name", None))
    return [n for n in names if n]

# tạo index nếu chưa có
existing = _list_index_names()
if INDEX_NAME not in existing:
    print(f"⏳ Chưa có index '{INDEX_NAME}', đang tạo...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PC_REGION),
    )
    # chờ index sẵn sàng (đơn giản)
    time.sleep(15)

index = pc.Index(INDEX_NAME)

# ---------- Embeddings ----------
def embed_many(texts: List[str]) -> List[List[float]]:
    # Azure OpenAI hỗ trợ batch input (mảng string) -> 1 call
    resp = client.embeddings.create(model=AZURE_EMBED_DEPLOY, input=texts)
    # bảo toàn thứ tự
    mapping = {d.index: d.embedding for d in resp.data}
    return [mapping[i] for i in range(len(texts))]

# ---------- CSV ----------
def _open_csv_auto(path: str):
    # đọc thử utf-8-sig, fallback utf-8
    try:
        return open(path, "r", encoding="utf-8-sig")
    except:
        return open(path, "r", encoding="utf-8")

csv_path = os.path.join(os.path.dirname(__file__), CSV_NAME)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Không thấy CSV: {csv_path}")

with _open_csv_auto(csv_path) as f:
    reader = csv.DictReader(f)
    fieldnames = [c.strip() for c in (reader.fieldnames or [])]

    # chấp nhận cả local_url hoặc url
    required_any = {
        "id", "name", "description", "price",  # luôn cần
        # url cột bắt buộc: ít nhất một trong hai
    }
    required_options = [{"local_url", "url"}]

    def _check_required():
        missing = required_any - set(fieldnames)
        if missing:
            raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing))}")
        ok_option = any(bool(set(opt) & set(fieldnames)) for opt in required_options)
        if not ok_option:
            raise ValueError("CSV thiếu cột URL: cần có 'local_url' hoặc 'url'.")

    _check_required()

    rows: List[Dict[str, Any]] = []
    for r in reader:
        # normalize keys & strip
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}

        _id = str(row.get("id", "")).strip()
        if not _id:
            continue  # bỏ dòng thiếu id

        # map URL
        url = (row.get("url") or row.get("local_url") or "").strip()

        # ép kiểu nhẹ price, category_id
        price_val = row.get("price")
        try:
            price = float(price_val) if price_val not in (None, "", "NULL") else None
        except:
            price = None

        cat_id_val = row.get("category_id")
        try:
            category_id = int(cat_id_val) if cat_id_val not in (None, "", "NULL") else None
        except:
            category_id = None

        item = {
            "id": _id,
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "description": row.get("description", ""),
            "price": price,
            "url": url,
            "image_url": row.get("image_url", ""),
            "category_id": category_id,
            "tags": row.get("tags", ""),
        }
        rows.append(item)

if not rows:
    print("⚠️ CSV trống hoặc không có dòng hợp lệ.")
    sys.exit(0)

# ---------- Chuẩn bị vectors ----------
def build_doc(x: Dict[str, Any]) -> str:
    parts = [
        x.get("name", ""),
        x.get("category", ""),
        x.get("tags", ""),
        x.get("description", ""),
        f"price {x['price']}" if x.get("price") is not None else "",
    ]
    return " | ".join([p for p in parts if p])

docs = [build_doc(x) for x in rows]

print(f"🧠 Tạo embeddings cho {len(docs)} mục...")
embs = []
# có thể chia nhỏ theo lô nếu CSV quá lớn; ở đây gọi 1 lần (Azure đã hỗ trợ batch)
# nếu cần chia lô: tách docs thành chunks rồi nối kết quả
try:
    embs = embed_many(docs)
except Exception as e:
    print(f"❌ Lỗi tạo embedding: {e}")
    sys.exit(1)

vectors = []
for it, vec in zip(rows, embs):
    vectors.append({
        "id": it["id"],
        "values": vec,
        "metadata": {
            "id": it["id"],
            "name": it["name"],
            "category": it["category"],
            "description": it["description"],
            "price": it["price"],
            "url": it["url"],
            "image_url": it["image_url"],
            "category_id": it["category_id"],
            "tags": it["tags"],
        }
    })

# ---------- Upsert ----------
total = len(vectors)
print(f"🚀 Bắt đầu nạp {total} sản phẩm vào Pinecone (namespace='{NAMESPACE}')...")
for i in range(0, total, BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(vectors=batch, namespace=NAMESPACE if NAMESPACE else None)
    print(f"✅ Upsert {min(i+BATCH_SIZE, total)}/{total}")

print(f"🎉 Hoàn tất nạp {total} dòng dữ liệu vào Pinecone!")
