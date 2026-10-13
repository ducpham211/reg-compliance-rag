"""Đánh giá tách riêng: (1) retrieval, (2) generation + từ chối. Ghi data/output/eval_report.md.

Chạy: PYTHONPATH=src python scripts/eval.py
"""
import json
import re
from pathlib import Path

import pandas as pd

from regrag.qa import ask
from regrag.retrieve import Index

DOCS = ("vbhn_86",)
gold = json.loads(Path("eval/golden.json").read_text(encoding="utf-8"))
ix = Index.from_processed()
rows = []
for g in gold:
    answerable = g.get("answerable", True)
    hits = ix.search(g["q"], k=6, docs=DOCS)
    ranked = [c.dieu for c, _, _ in hits]
    rank = next((i + 1 for i, d in enumerate(ranked) if d in g["dieu"]), None) if answerable else None
    a = ask(g["q"], ix, docs=DOCS)
    cited = {m for c in a.citations for m in re.findall(r"Điều (\w+)", c)}
    refused = a.status == "no_evidence"
    row = {"q": g["q"], "answerable": answerable, "rank": rank, "status": a.status, "answer": a.text, "cited": sorted(cited)}
    if answerable:
        row["cite_ok"] = bool(cited & set(g["dieu"]))
        row["kw_ok"] = all(k.lower() in a.text.lower() for k in g.get("kw", []))
        row["fail"] = ("retrieval" if rank is None else "refused_wrongly" if refused else
                       "wrong_citation" if not row["cite_ok"] else "wrong_answer" if not row["kw_ok"] else "")
    else:
        row["refused_ok"] = refused
        row["fail"] = "hallucinated" if not refused else ""
    rows.append(row)

df = pd.DataFrame(rows)
A, U = df[df.answerable], df[~df.answerable]
n = len(A)
md = ["# Báo cáo đánh giá\n", f"{n} câu có đáp án, {len(U)} câu không có căn cứ trong nguồn.\n",
      "## 1. Retrieval (độc lập với LLM)\n",
      f"- Hit@1: {(A['rank'] == 1).sum()}/{n}", f"- Hit@3: {(A['rank'].le(3)).sum()}/{n}", f"- Hit@6: {A['rank'].notna().sum()}/{n}",
      f"- MRR: {(1 / A['rank']).fillna(0).mean():.3f}\n",
      "## 2. Generation\n",
      f"- Trích dẫn đúng Điều: {A.cite_ok.sum()}/{n}", f"- Câu trả lời chứa ý chính (keyword): {A.kw_ok.sum()}/{n}",
      f"- Từ chối sai (có đáp án mà từ chối): {(A.fail == 'refused_wrongly').sum()}/{n}",
      f"- Từ chối đúng với câu không có căn cứ: {U.refused_ok.sum()}/{len(U)}\n",
      "## 3. Phân loại lỗi (lỗi nằm ở bước nào)\n", df[df.fail != ''].fail.value_counts().to_string() or "Không có lỗi", "\n",
      "## 4. Chi tiết các câu lỗi\n"]
for _, r in df[df.fail != ''].iterrows():
    md.append(f"- [{r.fail}] {r.q} -> rank={r['rank']} status={r.status} cited={r.cited} | {r.answer[:160]}")
Path("data/output").mkdir(parents=True, exist_ok=True)
Path("data/output/eval_report.md").write_text("\n".join(md), encoding="utf-8")
df.to_csv("data/output/eval_results.csv", index=False)
print("\n".join(md))
