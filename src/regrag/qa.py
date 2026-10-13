"""Hỏi đáp RAG: trả lời kèm citation lấy từ metadata chunk; từ chối khi không có căn cứ trong nguồn."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import llm
from .retrieve import Index, chunk_text, citation

# Ngưỡng từ chối sớm (không tốn request LLM). Hiệu chỉnh bằng bộ câu hỏi ở eval.
GATE_COS, GATE_BM25, WEAK_COS = 0.35, 8.0, 0.50

SYSTEM = (
    "Bạn là trợ lý tra cứu văn bản quy phạm pháp luật về hoạt động cho vay của tổ chức tín dụng. "
    "CHỈ được dùng thông tin trong các ĐOẠN TRÍCH được cung cấp. Không dùng kiến thức bên ngoài, không suy đoán. "
    "Nếu các đoạn trích không đủ để trả lời, đặt answerable=false và để answer rỗng. "
    "Trả lời ngắn gọn bằng tiếng Việt. Ghi id đoạn trích đã dùng vào cited_ids. Đây không phải tư vấn pháp lý."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "cited_ids"],
}
NO_EVIDENCE = "Không tìm thấy căn cứ trong các văn bản nguồn để trả lời câu hỏi này."


@dataclass
class Answer:
    status: str  # grounded | weak | no_evidence
    text: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[tuple[str, float]] = field(default_factory=list)


def ask(question: str, ix: Index, docs: tuple[str, ...] | None = ("vbhn_86",), k: int = 6) -> Answer:
    hits = ix.search(question, k=k, docs=docs)
    retrieved = [(citation(c), round(cos, 2)) for c, cos, _ in hits]
    top_cos, top_bm = max(h[1] for h in hits), max(h[2] for h in hits)
    if top_cos < GATE_COS and top_bm < GATE_BM25:
        return Answer("no_evidence", NO_EVIDENCE, [], retrieved)

    ctx = "\n\n".join(f"[{i}] ({citation(c)})\n{chunk_text(c)}" for i, (c, _, _) in enumerate(hits))
    res = llm.generate_json(f"ĐOẠN TRÍCH:\n{ctx}\n\nCÂU HỎI: {question}", SCHEMA, SYSTEM)
    ids = [i for i in res["cited_ids"] if 0 <= i < len(hits)]  # bỏ id bịa
    if not res["answerable"] or not res["answer"].strip() or not ids:
        return Answer("no_evidence", NO_EVIDENCE, [], retrieved)
    cites = [citation(hits[i][0]) for i in dict.fromkeys(ids)]
    return Answer("grounded" if top_cos >= WEAK_COS else "weak", res["answer"].strip(), cites, retrieved)
