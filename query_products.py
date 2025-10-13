# query_products.py  (Azure OpenAI + Pinecone search, hỗ trợ lọc Category & Giá)
import os, re
from typing import Optional, Tuple, Dict, Any, List
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

# --- ENV ---
AZURE_OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_EMBED_DEPLOY      = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

PINECONE_API_KEY        = os.getenv("PINECONE_API_KEY")
INDEX_NAME              = os.getenv("PINECONE_INDEX", "products-index")
NAMESPACE               = os.getenv("PINECONE_NAMESPACE", "products")  # có thể để trống
DEFAULT_TOP_K           = int(os.getenv("TOP_K", "10"))  # dùng 10 để liệt kê được nhiều

# --- Clients ---
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VER,
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Helpers ---
def embed(text: str) -> List[float]:
    r = client.embeddings.create(model=AZURE_EMBED_DEPLOY, input=text)
    return r.data[0].embedding

def pick_url(m: dict) -> str:
    return (m.get("url") or m.get("local_url") or "").strip()

def fmt_price(v):
    try:
        x = float(v)
        return f"{x:,.0f}".replace(",", ".")  # 1.234.567
    except:
        return str(v) if v is not None else "Chưa rõ"

# ---- Nhận diện Category (chuẩn hoá về metadata.category) ----
CATEGORY_MAP = {
    # key trong câu hỏi  -> giá trị category trong metadata
    "bình": "Bình",
    "binh": "Bình",
    "tranh": "Tranh",
    "hoa": "Hoa và Cây",
    "cây": "Hoa và Cây",
    "cay": "Hoa và Cây",
    "đồng hồ": "Đồng hồ",
    "dong ho": "Đồng hồ",
}

def detect_category(query: str) -> Optional[str]:
    q = query.lower()
    # kiểm tra cụm 2 từ trước
    if "đồng hồ" in q:
        return "Đồng hồ"
    # quét keys còn lại
    for k, v in CATEGORY_MAP.items():
        if k in q:
            return v
    return None

# ---- Parse giá từ câu hỏi ----
NUM_TOKEN = re.compile(
    r'(?P<num>(?:\d+[.,]?\d*)|(?:\d{1,3}(?:[.,]\d{3})+))\s*(?P<Unit>k|ng[aà]n|ngh[ìi]n|tr|tri[eệ]u|m)?',
    re.IGNORECASE
)

def to_number(token: str, unit: Optional[str]) -> Optional[float]:
    if not token:
        return None
    # chuẩn hoá số: "1.200.000" hoặc "1,200,000" -> 1200000
    # nhưng "0.5" vẫn cần giữ dấu chấm thập phân -> tạm thay dấu phẩy = chấm, xóa dấu chấm ngăn nghìn
    raw = token.replace(",", ".")
    # nếu có nhiều dấu chấm, giả định dấu chấm là ngăn nghìn -> bỏ hết, giữ 1 dấu cuối (khó phân biệt).
    # đơn giản: nếu có >1 dấu chấm và không có unit "tr" thì bỏ tất cả chấm
    if raw.count(".") > 1:
        raw = raw.replace(".", "")
    try:
        val = float(raw)
    except:
        return None
    unit = (unit or "").lower()
    if unit in ["k", "ngan", "ngàn", "nghìn", "nghin"]:
        val *= 1_000
    elif unit in ["tr", "triệu", "trieu", "m"]:  # chấp nhận "m" như million
        val *= 1_000_000
    return val

def parse_price_range(query: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Trả về (min_price, max_price)
    - "dưới 500k", "<= 500 ngàn" -> (None, 500000)
    - "trên 300k", ">= 300k" -> (300000, None)
    - "từ 300k đến 700k", "300k - 700k" -> (300000, 700000)
    - nếu chỉ 1 số mà không từ khoá -> coi như 'tối đa' (max)
    """
    q = query.lower()
    # range "từ X đến Y", "X - Y"
    m_range = re.search(r'từ\s+(.*?)\s+(?:đến|-)\s+(.*)', q)
    if m_range:
        nums = list(NUM_TOKEN.finditer(m_range.group(0)))
        if len(nums) >= 2:
            n1 = to_number(nums[0].group("num"), nums[0].group("Unit"))
            n2 = to_number(nums[1].group("num"), nums[1].group("Unit"))
            if n1 and n2:
                lo, hi = sorted([n1, n2])
                return lo, hi

    # dưới / <= / không quá / tối đa / max
    if any(k in q for k in ["dưới", "<=", "<", "không quá", "toi da", "tối đa", "max"]):
        m = NUM_TOKEN.search(q)
        if m:
            v = to_number(m.group("num"), m.group("Unit"))
            return None, v

    # trên / >= / ít nhất / tối thiểu / min
    if any(k in q for k in ["trên", ">=", ">", "ít nhất", "toi thieu", "tối thiểu", "min"]):
        m = NUM_TOKEN.search(q)
        if m:
            v = to_number(m.group("num"), m.group("Unit"))
            return v, None

    # chỉ 1 số -> coi như ngân sách tối đa
    nums = list(NUM_TOKEN.finditer(q))
    if len(nums) == 1:
        v = to_number(nums[0].group("num"), nums[0].group("Unit"))
        return None, v

    return None, None

# ---- Query Pinecone ----
def pinecone_query(query_vec: List[float], top_k: int, flt: Optional[Dict[str, Any]] = None):
    return index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE if NAMESPACE else None,
        filter=flt or None
    )

def print_match_list(matches: List[Dict[str, Any]], header: str):
    print(f"\n{header}")
    if not matches:
        print("  (Không có mục phù hợp)")
        return
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata") or {}
        name = md.get("name") or "(Không tên)"
        price = fmt_price(md.get("price"))
        url = pick_url(md)
        image_url = md.get("image_url") or ""
        cat = md.get("category") or ""
        score = float(m.get("score", 0.0))
        pid = md.get("id") or ""
        line = f"#{i}  [{cat}] {name} — {price} VNĐ (score {score:.3f})"
        if pid:
            line = f"#{i}  ID {pid} | " + line
        print(line)
        if url: print(f"     🔗 {url}")
        if image_url: print(f"     🖼  {image_url}")

# --- Input ---
query = input("🔎 Nhập mô tả sản phẩm bạn cần: ").strip()
if not query:
    print("❌ Bạn chưa nhập gì cả.")
    raise SystemExit

# --- Phân tích query ---
category = detect_category(query)           # e.g., "Bình"
min_price, max_price = parse_price_range(query)

# --- Tạo vector truy vấn ---
vec = embed(query)

# --- Nếu có category: liệt kê hết sản phẩm trong category (có thể kèm lọc giá) ---
filter_dict: Dict[str, Any] = {}
if category:
    filter_dict["category"] = category

# áp dụng filter giá nếu có
if min_price is not None or max_price is not None:
    price_filter: Dict[str, Any] = {}
    if min_price is not None:
        price_filter["$gte"] = float(min_price)
    if max_price is not None:
        price_filter["$lte"] = float(max_price)
    filter_dict["price"] = price_filter

had_filter = bool(filter_dict)

# --- Nếu có category hoặc có lọc giá => truy vấn với filter để LIỆT KÊ ---
if had_filter:
    # lấy nhiều một chút để có danh sách
    res_list = pinecone_query(vec, top_k=DEFAULT_TOP_K, flt=filter_dict)
    matches_list = res_list.get("matches") or []
    # In danh sách theo filter (category/price)
    title = "📂 Danh sách theo"
    parts = []
    if category: parts.append(f"loại: {category}")
    if min_price is not None and max_price is not None:
        parts.append(f"giá: {fmt_price(min_price)}–{fmt_price(max_price)} VNĐ")
    elif min_price is not None:
        parts.append(f"giá ≥ {fmt_price(min_price)} VNĐ")
    elif max_price is not None:
        parts.append(f"giá ≤ {fmt_price(max_price)} VNĐ")
    if not parts: parts.append("bộ lọc")
    print_match_list(matches_list, f"{title} " + ", ".join(parts))

# --- Sau đó: hiển thị SẢN PHẨM PHÙ HỢP NHẤT như code cũ ---
# Nếu đã có filter thì best-match nên trong phạm vi filter; nếu không có filter thì global
best_filter = filter_dict if had_filter else None
res_best = pinecone_query(vec, top_k=1, flt=best_filter)
best_matches = res_best.get("matches") or []

if not best_matches:
    # fallback: thử không filter nếu trước đó có filter mà rỗng
    if had_filter:
        res_best2 = pinecone_query(vec, top_k=1, flt=None)
        best_matches = res_best2.get("matches") or []

if not best_matches:
    print("\n❌ Không tìm thấy sản phẩm phù hợp.")
else:
    m = best_matches[0]
    md = m.get("metadata") or {}
    print("\n✅ Sản phẩm phù hợp nhất:")
    print(f"🏷️ Tên: {md.get('name', '(Không tên)')}")
    cat = md.get("category") or ""
    if cat: print(f"🗂  Loại: {cat}")
    print(f"💰 Giá: {fmt_price(md.get('price'))} VNĐ")
    print(f"🔗 Link: {pick_url(md)}")
    if md.get("image_url"):
        print(f"🖼 Hình ảnh: {md['image_url']}")
    print(f"📊 Độ tương đồng: {float(m.get('score', 0.0)):.3f}")
