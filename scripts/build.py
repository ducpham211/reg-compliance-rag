"""Chạy toàn bộ pipeline offline: parse -> so sánh -> giải thích/impact -> trích xuất yêu cầu -> xuất xlsx/json.

Kết quả LLM được cache (data/cache/llm) nên chạy lại sẽ tiếp tục từ chỗ dừng, không tốn quota thêm.
Chạy: PYTHONPATH=src python scripts/build.py [parse|compare|extract|rules|export|all]
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from regrag import analyze
from regrag.compare import compare
from regrag.parse import load, parse_ocr_pages, parse_vbhn, save

P = Path("data/processed")
OUT = Path("data/output")


def step_parse():
    save(parse_vbhn("docs/vbhn_86/86-vbhn-nhnn.pdf"), P / "vbhn_86.json")
    save(parse_ocr_pages("docs/tt39_2016_goc/pages"), P / "tt39_2016_goc.json")
    print("parse ok")


def step_compare():
    old, new = load(P / "tt39_2016_goc.json"), load(P / "vbhn_86.json")
    changes = compare(old, new)
    print(f"{len(changes)} thay đổi; gọi LLM giải thích theo từng Điều ...")
    analyze.summarize_changes(changes)
    rows = [{**asdict(c), "ref": c.ref} for c in changes]
    (P / "changes.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("compare ok")


def step_extract():
    cl = load(P / "vbhn_86.json")
    reqs = analyze.extract_requirements(cl)
    (P / "requirements.json").write_text(json.dumps(analyze.to_dicts(reqs), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(reqs)} yêu cầu")


def step_rules(batch: int = 30):
    reqs = [analyze.Requirement(**d) for d in json.loads((P / "requirements.json").read_text(encoding="utf-8"))]
    rules = []
    for i in range(0, len(reqs), batch):
        rules += analyze.generate_rules(reqs[i:i + batch])
    (P / "rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rules)} rules")


def step_export():
    """Xuất các file nộp: requirements.xlsx (kèm checklist), rules.json, compare.xlsx."""
    reqs = pd.DataFrame(json.loads((P / "requirements.json").read_text(encoding="utf-8")))
    checklist = pd.DataFrame({
        "ID": reqs.req_id, "Yêu cầu kiểm soát": reqs.action, "Chủ thể": reqs.subject, "Mức độ": reqs.obligation,
        "Bằng chứng / Kiểm soát": reqs.evidence_control, "Căn cứ": reqs.source, "Kết quả": "☐ Chưa kiểm",
    })
    reqs = reqs.rename(columns={"req_id": "ID", "dieu": "Điều", "khoan": "Khoản", "requirement": "Requirement", "subject": "Subject",
                                "action": "Action", "evidence_control": "Evidence/Control", "obligation": "Mức độ", "source": "Căn cứ"})
    with pd.ExcelWriter(OUT / "requirements_checklist.xlsx", engine="openpyxl") as w:
        reqs.to_excel(w, sheet_name="Requirements", index=False)
        checklist.to_excel(w, sheet_name="Checklist", index=False)
    rules = json.loads((P / "rules.json").read_text(encoding="utf-8"))
    (OUT / "rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    ch = pd.DataFrame(json.loads((P / "changes.json").read_text(encoding="utf-8")))
    cmp_df = pd.DataFrame({
        "Điều/Khoản": ch.ref, "Trạng thái": ch.status, "Độ giống": ch.similarity, "Bản gốc 2016": ch.old, "Bản hiện hành": ch.new,
        "Ý nghĩa thay đổi": ch.summary, "Thông tư sửa đổi": ch.amended_by,
        "Quy trình ảnh hưởng": ch.impact.map(lambda i: ", ".join((i or {}).get("processes", []))),
        "Bộ phận ảnh hưởng": ch.impact.map(lambda i: ", ".join((i or {}).get("departments", []))),
        "Trường dữ liệu ảnh hưởng": ch.impact.map(lambda i: ", ".join((i or {}).get("data_fields", []))),
    })
    cmp_df.to_excel(OUT / "compare_impact.xlsx", index=False)
    print("export ok ->", OUT)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("parse", step_parse), ("compare", step_compare), ("extract", step_extract), ("rules", step_rules), ("export", step_export)):
        if what in (name, "all"):
            fn()
