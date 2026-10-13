"""Parse văn bản pháp lý thành các chunk theo Điều/Khoản, giữ số trang và metadata sửa đổi."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

BODY_MIN_SIZE = 13  # VBHN: thân văn bản 14pt, chân trang 12pt, chỉ số chân trang < 12pt

RE_DIEU = re.compile(r"^Điều (\d+[a-z]?)\.\s*(.*)$")
RE_KHOAN = re.compile(r"^(\d{1,2})\s*[.,]\s*([^\d\s].*)$")  # OCR đôi khi đọc dấu chấm thành phẩy
RE_SECTION = re.compile(r"^(Chương|Mục)\s+[IVX\d]+", re.I)
RE_FIRST_DIEU = re.compile(r"^Điều 1\.\s*Phạm vi")


@dataclass
class Clause:
    doc_id: str
    dieu: str
    dieu_title: str
    khoan: str | None
    text: str
    page: int
    amended_by: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"Điều {self.dieu}" + (f" khoản {self.khoan}" if self.khoan else "")


def _vbhn_lines(pdf_path: Path):
    """Trả (body, notes). body: list (page, text, refs) cho dòng thân văn bản; notes: {số: nội dung chân trang}."""
    body, foot_lines = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            big = page.filter(lambda o: o["object_type"] != "char" or o["size"] >= BODY_MIN_SIZE)
            small = page.filter(lambda o: o["object_type"] == "char" and o["size"] < BODY_MIN_SIZE)
            b_lines = [ln for ln in big.extract_text_lines() if ln["text"].strip()]
            last_top = b_lines[-1]["top"] if b_lines else 40
            refs_by_line: dict[int, list[str]] = {}
            for ln in small.extract_text_lines():
                if ln["top"] <= 40:  # số trang
                    continue
                if ln["top"] > last_top + 5:
                    foot_lines.append(ln["text"].strip())
                else:  # chỉ số chân trang chen trong thân văn bản
                    i = min(range(len(b_lines)), key=lambda k: abs(b_lines[k]["top"] - ln["top"]))
                    refs_by_line.setdefault(i, []).extend(r for r in re.split(r"\D+", ln["text"]) if r)
            for i, ln in enumerate(b_lines):
                body.append((pno, ln["text"].strip(), refs_by_line.get(i, [])))
    return body, _split_footnotes(foot_lines)


def _split_footnotes(lines: list[str]) -> dict[str, str]:
    """Chân trang đánh số tuần tự 1,2,3...; dòng bắt đầu bằng số kế tiếp là footnote mới."""
    notes: dict[str, str] = {}
    nxt, cur = 1, None
    for ln in lines:
        m = re.match(rf"^{nxt}\s+(\S.*)$", ln)
        if m:
            cur = str(nxt)
            notes[cur] = m.group(1)
            nxt += 1
        elif cur:
            notes[cur] += " " + ln
    return notes


def _amended_by(refs: list[str], notes: dict[str, str]) -> list[str]:
    out = []
    for r in refs:
        for m in re.finditer(r"Thông tư số (\d+/\d{4}/TT-NHNN)", notes.get(r, "")):
            if m.group(1) not in out and m.group(1) != "39/2016/TT-NHNN":
                out.append(m.group(1))
    return out


def _clauses_from_lines(doc_id: str, lines, notes=None) -> list[Clause]:
    """lines: iterable (page, text, refs). Bắt đầu từ Điều 1 để bỏ phần căn cứ/lời dẫn."""
    notes = notes or {}
    clauses: list[Clause] = []
    started = False
    dieu = title = ""
    cur: Clause | None = None
    dieu_refs: list[str] = []
    skip_caps = False

    for page, text, refs in lines:
        if not started:
            if not RE_FIRST_DIEU.match(text):
                continue
            started = True
        m = RE_DIEU.match(text)
        if m:
            dieu, title = m.group(1), m.group(2).strip()
            dieu_refs = list(refs)
            cur = Clause(doc_id, dieu, title, None, "", page, _amended_by(dieu_refs, notes))
            clauses.append(cur)
            continue
        if cur is None:
            continue
        if RE_SECTION.match(text):  # "Chương III", "Mục 2" và dòng tiêu đề IN HOA ngay sau đó không thuộc Điều nào
            skip_caps = True
            continue
        if skip_caps and text.upper() == text and not RE_KHOAN.match(text):
            continue
        skip_caps = False
        k = RE_KHOAN.match(text)
        if k and int(k.group(1)) == (int(cur.khoan) + 1 if cur.khoan else 1):
            # dấu sửa đổi ở tiêu đề Điều áp dụng cho cả Điều; dấu ở khoản chỉ cho khoản đó
            cur = Clause(doc_id, dieu, title, k.group(1), text, page,
                         sorted(set(_amended_by(list(refs), notes)) | set(_amended_by(dieu_refs, notes))))
            clauses.append(cur)
            continue
        cur.text = (cur.text + " " + text).strip()
        if refs:
            cur.amended_by = sorted(set(cur.amended_by) | set(_amended_by(refs, notes)))
    # chunk tiêu đề Điều rỗng mà Điều có khoản thì bỏ (câu dẫn nhập, nếu có, vẫn nằm trong chunk tiêu đề)
    has_khoan = {c.dieu for c in clauses if c.khoan}
    return [c for c in clauses if c.text or c.dieu not in has_khoan]


def parse_vbhn(pdf_path: str | Path, doc_id: str = "vbhn_86") -> list[Clause]:
    body, notes = _vbhn_lines(Path(pdf_path))
    return _clauses_from_lines(doc_id, body, notes)


def parse_ocr_pages(pages_dir: str | Path, doc_id: str = "tt39_2016_goc") -> list[Clause]:
    """Văn bản gốc là ảnh scan đã OCR: mỗi trang một file p-XX.txt."""
    lines = []
    for f in sorted(Path(pages_dir).glob("p-*.txt")):
        page = int(re.search(r"p-(\d+)", f.name).group(1))
        for raw in f.read_text(encoding="utf-8").splitlines():
            t = re.sub(r"^[LlI]([.,]\s)", r"1\1", raw.strip())  # OCR hay đọc "1." thành "L." / "I."
            if t:
                lines.append((page, t, []))
    return _clauses_from_lines(doc_id, lines)


def save(clauses: list[Clause], out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps([asdict(c) for c in clauses], ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: str | Path) -> list[Clause]:
    return [Clause(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
