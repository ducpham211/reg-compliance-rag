"""Các tác vụ dùng LLM: giải thích thay đổi + impact map, trích xuất yêu cầu, sinh rules máy đọc được."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from . import llm
from .compare import Change, _dieu_key, word_diff
from .parse import Clause
from .retrieve import citation

SKIP_EXTRACT = {"1", "2", "29", "32", "32c", "32g", "32h", "33", "35"}  # phạm vi, định nghĩa, đã bãi bỏ, hiệu lực/tổ chức thực hiện

# Gợi ý cho impact map (LLM được phép trả rỗng nếu không liên quan)
PROCESSES = ["Tiếp nhận hồ sơ vay", "Nhận biết/xác minh khách hàng", "Thẩm định", "Phê duyệt cho vay", "Ký thỏa thuận cho vay",
             "Giải ngân", "Thu nợ gốc/lãi", "Cơ cấu lại thời hạn trả nợ", "Kiểm tra sử dụng vốn vay", "Công khai lãi suất/phí",
             "Quản lý rủi ro", "Lưu trữ hồ sơ/dữ liệu", "Hệ thống thông tin/an toàn thông tin"]
DEPARTMENTS = ["Kinh doanh/Quan hệ khách hàng", "Thẩm định tín dụng", "Quản lý rủi ro", "Pháp chế/Tuân thủ",
               "Vận hành/Giải ngân", "Công nghệ thông tin", "Kế toán/Tài chính", "Xử lý nợ"]
DATA_FIELDS = ["age", "monthly_income_mil", "loan_amount_mil", "loan_term_months", "credit_score",
               "employment_type", "loan_purpose", "default_risk"]  # các cột của data_raw.xlsx

CHANGE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "idx": {"type": "integer"}, "summary": {"type": "string"},
        "processes": {"type": "array", "items": {"type": "string"}},
        "departments": {"type": "array", "items": {"type": "string"}},
        "data_fields": {"type": "array", "items": {"type": "string"}}},
        "required": ["idx", "summary", "processes", "departments", "data_fields"]}}},
    "required": ["items"],
}
REQ_SCHEMA = {
    "type": "object",
    "properties": {"requirements": {"type": "array", "items": {"type": "object", "properties": {
        "dieu": {"type": "string"}, "khoan": {"type": "string"}, "requirement": {"type": "string"}, "subject": {"type": "string"},
        "action": {"type": "string"}, "evidence_control": {"type": "string"},
        "obligation": {"type": "string", "enum": ["bắt buộc", "cấm", "cho phép/điều kiện"]}},
        "required": ["dieu", "khoan", "requirement", "subject", "action", "evidence_control", "obligation"]}}},
    "required": ["requirements"],
}
RULE_SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {"type": "object", "properties": {
        "req_id": {"type": "string"}, "applies_to": {"type": "string"}, "condition": {"type": "string"},
        "check": {"type": "string"}, "field": {"type": "string"},
        "operator": {"type": "string", "enum": ["==", "!=", "<=", ">=", "<", ">", "in", "not_in", "exists", "manual"]},
        "value": {"type": "string"}, "severity": {"type": "string", "enum": ["must", "must_not", "may"]}},
        "required": ["req_id", "applies_to", "condition", "check", "field", "operator", "value", "severity"]}}},
    "required": ["rules"],
}


# ---------- 1. Giải thích thay đổi + impact map ----------
def _batches(items: list[tuple[str, int]], budget: int) -> list[list[int]]:
    """Gom chỉ số item vào các batch có tổng độ dài <= budget (mỗi batch ít nhất 1 item)."""
    out, cur, size = [], [], 0
    for i, (_, n) in enumerate(items):
        if cur and size + n > budget:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    return out + ([cur] if cur else [])


def summarize_changes(changes: list[Change], budget: int = 9000) -> list[Change]:
    system = ("Bạn là chuyên viên tuân thủ ngân hàng. Với mỗi thay đổi giữa bản gốc 2016 và bản hợp nhất hiện hành của Thông tư 39/2016/TT-NHNN, "
              "viết 1-2 câu tiếng Việt nêu ý nghĩa thực tế của thay đổi (ai phải làm gì khác đi). Chỉ dựa vào văn bản được đưa. "
              "Với impact: chọn từ danh sách gợi ý các quy trình, bộ phận, trường dữ liệu BỊ ẢNH HƯỞNG TRỰC TIẾP; để mảng rỗng nếu không có căn cứ. "
              "idx phải trùng với số thứ tự trong ngoặc vuông.")
    blocks = []
    for i, c in enumerate(changes):
        body = word_diff(c.old, c.new) if c.old and c.new else (c.new or c.old)
        blocks.append(f"[{i}] {c.status} - {c.ref}\n{body[:1500]}")
    head = (f"Gợi ý quy trình: {PROCESSES}\nGợi ý bộ phận: {DEPARTMENTS}\nGợi ý trường dữ liệu: {DATA_FIELDS}\n"
            "Quy ước diff: ~~chữ bị xóa~~, **chữ thêm mới**.\n\n")
    for idxs in _batches([(b, len(b)) for b in blocks], budget):
        res = llm.generate_json(head + "\n\n".join(blocks[i] for i in idxs), CHANGE_SCHEMA, system)
        for it in res["items"]:
            if it["idx"] in idxs:
                c = changes[it["idx"]]
                c.summary = it["summary"].strip()
                c.impact = {"processes": it["processes"], "departments": it["departments"], "data_fields": it["data_fields"]}
    return changes


# ---------- 2. Trích xuất yêu cầu ----------
@dataclass
class Requirement:
    req_id: str
    dieu: str
    khoan: str
    requirement: str
    subject: str
    action: str
    evidence_control: str
    obligation: str
    source: str  # citation lấy từ metadata, không do LLM sinh


def extract_requirements(clauses: list[Clause], dieu_list: list[str] | None = None, budget: int = 7000) -> list[Requirement]:
    by: dict[str, list[Clause]] = {}
    for c in clauses:
        by.setdefault(c.dieu, []).append(c)
    system = ("Bạn trích xuất YÊU CẦU TUÂN THỦ từ điều khoản pháp lý. Chỉ trích các câu quy phạm (nghĩa vụ, cấm, điều kiện), không trích định nghĩa hay lời dẫn. "
              "Mỗi yêu cầu là một ý độc lập, diễn đạt lại ngắn gọn, KHÔNG thêm thông tin ngoài văn bản. "
              "subject = ai phải thực hiện; action = hành động phải làm/không được làm; "
              "evidence_control = hồ sơ, bằng chứng hoặc biện pháp kiểm soát để chứng minh tuân thủ (nếu văn bản không nêu rõ, viết 'Không nêu rõ trong văn bản'). "
              "dieu = số Điều (vd '8', '32a'); khoan = số khoản chứa yêu cầu (rỗng nếu Điều không chia khoản).")
    todo = [d for d in (dieu_list or list(by)) if d in by and d not in SKIP_EXTRACT and any(c.text for c in by[d])]
    texts = {d: f"### Điều {d}. {by[d][0].dieu_title}\n" + "\n".join(f"[khoản {c.khoan or '-'}] {c.text}" for c in by[d] if c.text) for d in todo}
    out: list[Requirement] = []
    counter: dict[str, int] = {}
    groups = [[todo[i] for i in idxs] for idxs in _batches([(d, len(texts[d])) for d in todo], budget)]
    retried = False
    while groups:
        group = groups.pop(0)
        res = llm.generate_json("\n\n".join(texts[d] for d in group), REQ_SCHEMA, system)
        for r in res["requirements"]:
            d = re.sub(r"[^0-9a-z]", "", r["dieu"].lower().replace("điều", ""))
            if d not in group:
                continue  # Điều bịa/không thuộc batch
            cl = by[d]
            k = re.sub(r"\D", "", r["khoan"]) if r["khoan"] else ""
            src = {c.khoan or "": c for c in cl}.get(k) or cl[0]  # khoản bịa -> lùi về Điều; citation luôn lấy từ metadata
            counter[d] = counter.get(d, 0) + 1
            out.append(Requirement(f"R-{d}-{counter[d]}", d, src.khoan or "", r["requirement"].strip(), r["subject"].strip(),
                                   r["action"].strip(), r["evidence_control"].strip(), r["obligation"], citation(src)))
        if not groups and not retried:  # model nhẹ hay bỏ sót Điều ngắn khi gom batch: thử lại riêng từng Điều bị sót
            retried = True
            groups = [[d] for d in todo if d not in counter]
    return sorted(out, key=lambda r: (_dieu_key(r.dieu), r.req_id))


# ---------- 3. Rules máy đọc được ----------
def generate_rules(reqs: list[Requirement]) -> list[dict]:
    system = ("Chuyển các yêu cầu tuân thủ thành rule máy đọc được. Mỗi rule: applies_to (đối tượng), condition (điều kiện áp dụng), "
              "check (phép kiểm tra bằng lời), field (tên trường dữ liệu cần kiểm tra, snake_case; nếu chỉ kiểm tra thủ công dùng 'n/a'), "
              "operator, value, severity (must/must_not/may). Dùng operator 'manual' khi không thể kiểm tra tự động từ dữ liệu. "
              "Chỉ dùng thông tin trong yêu cầu, không bịa số liệu. req_id phải trùng với id đầu vào.")
    prompt = "\n".join(f"{r.req_id}: [{r.obligation}] {r.subject} - {r.action}. Căn cứ: {r.source}" for r in reqs)
    rules = llm.generate_json(prompt, RULE_SCHEMA, system)["rules"]
    src = {r.req_id: r.source for r in reqs}
    return [{**r, "source": src[r["req_id"]]} for r in rules if r["req_id"] in src]


def to_dicts(items) -> list[dict]:
    return [asdict(i) for i in items]
