"""So sánh hai phiên bản: căn chỉnh theo Điều rồi theo khoản, phân loại Added/Removed/Modified."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .parse import Clause
from .retrieve import fold

SAME, MATCH = 0.985, 0.55  # SAME: coi như không đổi (đã bỏ dấu nên lỗi dấu OCR không ảnh hưởng); MATCH: đủ giống để ghép cặp khoản
REPEAL = re.compile(r"được bãi bỏ|bãi bỏ", re.I)


@dataclass
class Change:
    dieu: str
    khoan_old: str | None
    khoan_new: str | None
    status: str  # Added | Removed | Modified
    old: str
    new: str
    similarity: float
    page_old: int | None = None
    page_new: int | None = None
    amended_by: str = ""
    summary: str = ""  # LLM điền sau
    impact: dict | None = None  # LLM điền sau: quy trình, bộ phận, trường dữ liệu bị ảnh hưởng

    @property
    def ref(self) -> str:
        k = self.khoan_new or self.khoan_old
        return f"Điều {self.dieu}" + (f" khoản {k}" if k else "")


def _norm(t: str) -> str:
    t = re.sub(r"^\d{1,2}[.,]\s+", "", t.strip())  # bỏ số khoản: đánh số lại không phải thay đổi nội dung
    return re.sub(r"\s+", " ", fold(t)).strip()


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b), autojunk=False).ratio()


def _dieu_key(d: str):
    m = re.match(r"(\d+)([a-z]?)", d)
    return int(m.group(1)), m.group(2)


def _by_dieu(clauses: list[Clause]) -> dict[str, list[Clause]]:
    out: dict[str, list[Clause]] = {}
    for c in clauses:
        out.setdefault(c.dieu, []).append(c)
    return out


def _chunk_view(c: Clause) -> str:
    return c.text or c.dieu_title


def compare(old: list[Clause], new: list[Clause]) -> list[Change]:
    o, n = _by_dieu(old), _by_dieu(new)
    changes: list[Change] = []
    for d in sorted(set(o) | set(n), key=_dieu_key):
        oc, nc = o.get(d, []), n.get(d, [])
        if not oc:
            for c in nc:
                if REPEAL.search(_chunk_view(c)):  # Điều mới tạo rồi bãi bỏ: không có nội dung để báo
                    continue
                changes.append(Change(d, None, c.khoan, "Added", "", _chunk_view(c), 0.0, None, c.page, ", ".join(c.amended_by)))
            continue
        if not nc:
            for c in oc:
                changes.append(Change(d, c.khoan, None, "Removed", _chunk_view(c), "", 0.0, c.page, None))
            continue
        changes += _compare_dieu(d, oc, nc)
    return changes


def _compare_dieu(d: str, oc: list[Clause], nc: list[Clause]) -> list[Change]:
    """Ghép tham lam theo độ giống cao nhất giữa các chunk (khoản) của cùng một Điều."""
    pairs = sorted(((_sim(_chunk_view(a), _chunk_view(b)), i, j) for i, a in enumerate(oc) for j, b in enumerate(nc)), reverse=True)
    used_o, used_n, matched = set(), set(), []
    for s, i, j in pairs:
        if s < MATCH or i in used_o or j in used_n:
            continue
        used_o.add(i), used_n.add(j), matched.append((i, j, s))
    # Vòng 2: khoản cùng số mà độ giống thấp (viết lại mạnh hoặc bị bãi bỏ) vẫn là một thay đổi Modified/Removed, không phải Removed + Added
    for i, a in enumerate(oc):
        if i in used_o:
            continue
        for j, b in enumerate(nc):
            if j not in used_n and a.khoan == b.khoan and (a.khoan or len(oc) == len(nc) == 1 or not a.khoan):
                used_o.add(i), used_n.add(j), matched.append((i, j, _sim(_chunk_view(a), _chunk_view(b))))
                break
    # Vòng 3: bản cũ chỉ có một đoạn cấp Điều (không chia khoản) còn bản mới chia khoản -> một thay đổi Modified cho cả Điều
    left_o = [i for i in range(len(oc)) if i not in used_o]
    left_n = [j for j in range(len(nc)) if j not in used_n]
    merged: Change | None = None
    if len(left_o) == 1 and left_n and oc[left_o[0]].khoan is None:
        a = oc[left_o[0]]
        new_text = " ".join(_chunk_view(nc[j]) for j in left_n)
        merged = Change(d, None, None, "Modified", _chunk_view(a), new_text, round(_sim(_chunk_view(a), new_text), 3),
                        a.page, nc[left_n[0]].page, ", ".join(sorted({x for j in left_n for x in nc[j].amended_by})))
        used_o.add(left_o[0]), used_n.update(left_n)
    out: list[Change] = [merged] if merged else []
    for i, j, s in matched:
        a, b = oc[i], nc[j]
        if s >= SAME:
            continue
        status = "Removed" if REPEAL.search(b.dieu_title + " " + b.text) and not REPEAL.search(a.text) else "Modified"
        out.append(Change(d, a.khoan, b.khoan, status, _chunk_view(a), _chunk_view(b), round(s, 3), a.page, b.page, ", ".join(b.amended_by)))
    for i, a in enumerate(oc):
        if i not in used_o:
            out.append(Change(d, a.khoan, None, "Removed", _chunk_view(a), "", 0.0, a.page, None))
    for j, b in enumerate(nc):
        if j not in used_n and not REPEAL.search(_chunk_view(b)):  # tiêu đề "(được bãi bỏ)" không phải nội dung mới
            out.append(Change(d, None, b.khoan, "Added", "", _chunk_view(b), 0.0, None, b.page, ", ".join(b.amended_by)))
    return out


def word_diff(old: str, new: str) -> str:
    """Diff mức từ dạng markdown: ~~xóa~~ và **thêm**."""
    a, b = old.split(), new.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(" ".join(a[i1:i2]))
        else:
            if i2 > i1:
                out.append("~~" + " ".join(a[i1:i2]) + "~~")
            if j2 > j1:
                out.append("**" + " ".join(b[j1:j2]) + "**")
    return " ".join(out)
