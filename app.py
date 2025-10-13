# app.py — FastAPI + Azure OpenAI + Pinecone (lọc loại & giá + ưu tiên khớp tên an toàn)
import os, re, unicodedata
from typing import Optional, Dict, Any

from fastapi import FastAPI, Query, Request, HTTPException
from dotenv import load_dotenv
from openai import AzureOpenAI, APIStatusError, APITimeoutError, APIConnectionError
from pinecone import Pinecone
from difflib import SequenceMatcher
COMPACT_CARD = os.getenv("COMPACT_CARD", "1") == "1"   # bật mặc định
# ----------------- Init -----------------
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
    timeout=30.0,
    max_retries=2,
)
EMBED_DEPLOY = os.getenv("AZURE_EMBED_DEPLOY", "embedding-deploy")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "products-index")
index = pc.Index(INDEX_NAME)

app = FastAPI()

# ----------------- Helpers -----------------
VN_CATEGORY_ALIASES: Dict[str, set] = {
    "Bình": {"binh", "bình", "lọ", "lo", "vase"},
    "Tranh": {"tranh", "khung tranh", "poster", "canvas"},
    "Hoa và Cây": {"hoa", "cành", "canh", "cây", "cay", "flower", "plant"},
    "Đồng hồ": {"đồng hồ", "dong ho", "clock"},
}

# Stopwords/fillers khi chat
STOPWORDS = {
    "tu", "tư", "van", "vấn", "tu van",
    "tim", "tìm", "giup", "giúp", "xem", "mua",
    "cho", "cho minh", "mình", "minh", "can", "cần",
    "hoi", "hỏi", "gi", "gì", "di", "đi", "cai", "cái",
    "tu van giup", "goi y", "gợi ý", "tu van mua", "tu van san pham",
}

def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s.lower()).strip()

def detect_category(q: str) -> Optional[str]:
    nq = normalize_text(q)
    for cat, aliases in VN_CATEGORY_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", nq):
                return cat
    return None

def parse_price_number(token: str) -> Optional[int]:
    """
    '500', '500k', '500 ngàn', '500 nghìn'  -> 500_000
    '1tr', '1 triệu', '1m'                 -> 1_000_000
    '700.000', '700,000'                   -> 700_000
    """
    if token is None:
        return None
    t = token.lower().strip()
    t = t.replace(".", "").replace(",", "")

    if t.endswith("k"):
        num = re.sub(r"k$", "", t)
        return int(num) * 1000 if num.isdigit() else None

    if re.search(r"(ngan|ngàn|nghìn|nghin)$", t):
        num = re.sub(r"(ngan|ngàn|nghìn|nghin)$", "", t).strip()
        return int(num) * 1000 if num.isdigit() else None

    if re.search(r"(tr|triệu|trieu|m)$", t):
        num = re.sub(r"(tr|triệu|trieu|m)$", "", t).strip()
        try:
            return int(float(num) * 1_000_000)
        except Exception:
            return None

    if t.isdigit():
        n = int(t)
        if n < 10_000:  # 500 -> 500k
            return n * 1000
        return n

    return None

def extract_price_filters(q: str) -> Dict[str, int]:
    """
    Trả về {'max': x} hoặc {'min': x} hoặc {'min': a, 'max': b}
    Hiểu: 'dưới 500k', 'trên 800k', 'từ 500 đến 800', '700-900k', '500k', '500 ngàn'
    """
    nq = normalize_text(q)

    # Khoảng A - B
    m = re.search(
        r"(\d[\d\.,]*\s*(?:k|ngan|ngàn|nghìn|tr|triệu|trieu|m)?)\s*(?:-|đến|to)\s*(\d[\d\.,]*\s*(?:k|ngan|ngàn|nghìn|tr|triệu|trieu|m)?)",
        nq,
    )
    if m:
        a = parse_price_number(m.group(1))
        b = parse_price_number(m.group(2))
        if a and b and a <= b:
            return {"min": a, "max": b}

    # dưới / <=
    m = re.search(r"(dưới|<=|<|ít hơn|nhỏ hơn)\s*(\d[\d\.,]*\s*(?:k|ngan|ngàn|nghìn|tr|triệu|trieu|m)?)", nq)
    if m:
        val = parse_price_number(m.group(2))
        if val:
            return {"max": val}

    # trên / >=
    m = re.search(r"(trên|>=|>|nhiều hơn|lớn hơn|tối thiểu|ít nhất)\s*(\d[\d\.,]*\s*(?:k|ngan|ngàn|nghìn|tr|triệu|trieu|m)?)", nq)
    if m:
        val = parse_price_number(m.group(2))
        if val:
            return {"min": val}

    # Chỉ 1 con số → hiểu là "tối đa"
    m = re.search(r"(\d[\d\.,]*\s*(?:k|ngan|ngàn|nghìn|tr|triệu|trieu|m)?)", nq)
    if m:
        val = parse_price_number(m.group(1))
        if val:
            return {"max": val}

    return {}

def build_filter(category: Optional[str], price_filter: Dict[str, int]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if category:
        f["category"] = {"$eq": category}
    if "min" in price_filter and "max" in price_filter:
        f["price"] = {"$gte": price_filter["min"], "$lte": price_filter["max"]}
    elif "min" in price_filter:
        f["price"] = {"$gte": price_filter["min"]}
    elif "max" in price_filter:
        f["price"] = {"$lte": price_filter["max"]}
    return f

# ----- Tokens loại (chuẩn hoá) -----
def _collect_category_tokens():
    toks = set()
    for cat, aliases in VN_CATEGORY_ALIASES.items():
        toks.add(normalize_text(cat))
        for a in aliases:
            toks.add(normalize_text(a))
    # thêm phiên bản bỏ dấu
    more = set()
    for t in toks:
        more.add(re.sub(r"\s+", " ", "".join(
            ch for ch in unicodedata.normalize("NFD", t) if not unicodedata.combining(ch)
        )))
    toks |= more
    toks = {re.sub(r"\s+", " ", t.strip()) for t in toks}
    return toks

CATEGORY_TOKENS = _collect_category_tokens()

def is_generic_query(q: str) -> bool:
    nq = normalize_text(q)
    if not nq:
        return True
    toks = nq.split()
    # bỏ stopwords
    toks = [t for t in toks if t not in STOPWORDS]
    if not toks:
        return True
    # nếu mọi token còn lại đều là token loại
    if all(t in CATEGORY_TOKENS for t in toks):
        return True
    # một token rất ngắn và là token loại
    if len(toks) == 1 and len(toks[0]) <= 5 and toks[0] in CATEGORY_TOKENS:
        return True
    return False

# ----- So khớp tên -----
def strip_accents(s: str) -> str:
    s = "" if s is None else str(s)
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))

def norm_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = strip_accents(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def name_match_score(q: str, name: str) -> float:
    """Điểm giống nhau giữa câu hỏi và tên sp (0..1)."""
    qn, nn = norm_name(q), norm_name(name)
    if not qn or not nn:
        return 0.0

    qn_tokens = qn.split()
    # chỉ cộng base cao nếu query đủ đặc trưng (>= 2 token hoặc dài >= 6)
    if (qn in nn or nn in qn) and (len(qn_tokens) >= 2 or len(qn) >= 6):
        base = 0.95
    else:
        base = 0.0

    seq = SequenceMatcher(None, qn, nn).ratio()
    qtok, ntok = set(qn_tokens), set(nn.split())
    jacc = (len(qtok & ntok) / len(qtok | ntok)) if (qtok and ntok) else 0.0

    return max(base, (seq * 0.6 + jacc * 0.4))

# ----- Embedding wrapper -----
def embed(text: str):
    try:
        r = client.embeddings.create(model=EMBED_DEPLOY, input=text)
        return r.data[0].embedding
    except (APITimeoutError, APIConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Embedding timeout/connection: {e.__class__.__name__}: {str(e)}")
    except APIStatusError as e:
        msg = f"Azure API error ({e.status_code}): {getattr(e, 'message', str(e))}"
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e.__class__.__name__}: {str(e)}")

def format_item(m: Dict[str, Any], score: Optional[float] = None) -> Dict[str, Any]:
    return {
        "name": m.get("name", ""),
        "category": m.get("category", ""),
        "price": m.get("price", ""),
        "url": m.get("url", ""),
        "image_url": m.get("image_url", ""),
        "score": score,
    }

# ----------------- Core search -----------------
def smart_search(q: str, top_k_category: int = 30, top_k_best: int = 10) -> Dict[str, Any]:
    # 1) Phân tích loại + giá để làm filter
    cat = detect_category(q)
    pf = extract_price_filters(q)
    pine_filter = build_filter(cat, pf)

    # 2) Lấy ứng viên theo ngữ nghĩa
    vec_q = embed(q)
    res_best = index.query(vector=vec_q, top_k=top_k_best, include_metadata=True, filter=pine_filter)
    matches = res_best.get("matches", [])

    # 3) Chấm điểm "độ giống tên"
    best_by_name = None
    best_name_score = 0.0
    alts = []
    for r in matches:
        m = r["metadata"]
        try:
            score_n = name_match_score(q, m.get("name", ""))
        except Exception:
            score_n = 0.0  # an toàn
        item = format_item(m, r.get("score"))
        item["name_match"] = round(score_n, 3)
        if score_n > best_name_score:
            best_name_score = score_n
            best_by_name = item
        alts.append(item)

    STRONG_NAME_THRESHOLD = 0.85
    strong_name_hit = bool(best_by_name and best_name_score >= STRONG_NAME_THRESHOLD)

    category_results = []
    best = None

    # 4) Chỉ ưu tiên tên khi KHÔNG phải query chung chung
    if strong_name_hit and not is_generic_query(q):
        best = best_by_name
        alternatives = []  # không trả list phụ khi trùng tên mạnh
    else:
        # Có loại → liệt kê theo loại (kết hợp lọc giá nếu có)
        if cat:
            vec_cat = embed(cat)
            res_cat = index.query(vector=vec_cat, top_k=top_k_category, include_metadata=True, filter=pine_filter)
            for r in res_cat.get("matches", []):
                category_results.append(format_item(r["metadata"], r.get("score")))

        # best theo vector
        if matches:
            r0 = matches[0]
            best = format_item(r0["metadata"], r0.get("score"))
        alternatives = [format_item(r["metadata"], r.get("score")) for r in matches[1:5]]

    return {
        "category": cat,
        "price_filter": pf,
        "category_results": category_results,
        "best_match": best,
        "alternatives": alternatives,
        "name_match_score": round(best_name_score, 3) if best_by_name else 0.0,
        "used_name_priority": bool(strong_name_hit and not is_generic_query(q)),
    }

# ----------------- Endpoints -----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/diag")
def diag():
    vec = embed("hello")
    _ = index.describe_index_stats()
    return {
        "embed_len": len(vec),
        "embed_deploy": EMBED_DEPLOY,
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "pinecone_index": INDEX_NAME
    }

@app.get("/search")
def search(q: str = Query(..., description="Câu hỏi/mô tả sản phẩm")):
    try:
        return smart_search(q)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e.__class__.__name__}")

@app.post("/search")
async def search_webhook(request: Request):
    body = await request.json()
    query = body.get("queryResult", {}).get("queryText", "")

    if not query:
        return {"fulfillmentText": "Mình chưa hiểu bạn muốn tìm gì 🧐"}

    result = smart_search(query)

    title_best = result["best_match"]["name"] if result["best_match"] else "Không tìm thấy sản phẩm phù hợp."
    subtitle_best = ""
    if result["best_match"] and result["best_match"].get("price") not in (None, ""):
        try:
            subtitle_best = f"Giá: {int(float(result['best_match']['price'])):,} VND".replace(",", ".")
        except Exception:
            subtitle_best = ""

    blocks = []
    # Nếu có list theo loại -> render trước
    if result["category_results"]:
        header = {
            "type": "info",
            "title": f"Kết quả theo loại: {result['category']}",
            "subtitle": f"Tìm thấy {len(result['category_results'])} sản phẩm"
        }
        blocks.append(header)

        for item in result["category_results"][:10]:
            price_text = ""
            if item.get("price") not in ("", None):
                try:
                    price_text = f"Giá: {int(float(item['price'])):,} VND".replace(",", ".")
                except Exception:
                    price_text = ""
            
            # ✅ Dạng thumbnail: ảnh nhỏ nằm trong card
            blocks.append({
                "type": "card",
                "title": item.get("name", ""),
                "subtitle": price_text,
                "image": { "src": { "rawUrl": item.get("image_url") or "" } },
                "actionLink": item.get("url", "")
            })

    # Best match (gợi ý phù hợp nhất)
    if result["best_match"]:
        price_text = ""
        if result["best_match"].get("price") not in (None, ""):
            try:
                price_text = f"Giá: {int(float(result['best_match']['price'])):,} VND".replace(",", ".")
            except Exception:
                price_text = ""

        blocks += [
            {"type": "divider"},
            {"type": "info", "title": "Gợi ý phù hợp nhất", "subtitle": ""},
            {
                # ✅ Thumbnail gọn, không còn ảnh full
                "type": "card",
                "title": result["best_match"].get("name", ""),
                "subtitle": price_text,
                "image": { "src": { "rawUrl": result["best_match"].get("image_url") or "" } },
                "actionLink": result["best_match"].get("url") or ""
            },
            {
                "type": "chips",
                "options": [
                    { "text": "Xem chi tiết", "link": result["best_match"].get("url") or "" }
                ]
            }
        ]

    # Gói payload gửi lại cho Dialogflow Messenger
    payload = {"richContent": [blocks]} if blocks else None

    return {
        "fulfillmentText": f"{title_best}\n{subtitle_best}\n{(result['best_match'] or {}).get('url','')}",
        "fulfillmentMessages": [{"payload": payload}] if payload else [],
    }
