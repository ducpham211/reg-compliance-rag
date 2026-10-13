"""Regulation & Compliance Intelligence Platform - demo Gradio.

Chạy: PYTHONPATH=src python app.py   (cần `scripts/build.py` đã chạy để có data/processed/*)
"""
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from regrag.compare import word_diff
from regrag.qa import ask
from regrag.retrieve import DOC_LABEL, Index

P, OUT = Path("data/processed"), Path("data/output")
DISCLAIMER = "⚠️ Bản demo học tập trên văn bản công khai (Thông tư 39/2016/TT-NHNN). Không phải tư vấn pháp lý."
SCOPES = {"Bản hiện hành (VBHN 86)": ("vbhn_86",), "Bản gốc 2016": ("tt39_2016_goc",), "Cả hai": ("vbhn_86", "tt39_2016_goc")}
BADGE = {"grounded": "🟢 **Có căn cứ**", "weak": "🟡 **Căn cứ yếu** - nên đối chiếu nguyên văn", "no_evidence": "🔴 **Không có căn cứ trong nguồn**"}

_index: Index | None = None


def index() -> Index:
    global _index
    if _index is None:
        _index = Index.from_processed()
    return _index


def _json(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


REQS = pd.DataFrame(_json("requirements.json"))
RULES = pd.DataFrame(_json("rules.json"))
CHANGES = _json("changes.json")
for i, c in enumerate(CHANGES):
    c["id"] = i


def _tmp(name: str) -> str:
    return str(Path(tempfile.mkdtemp()) / name)


# ---------- Tab 1: hỏi đáp ----------
def do_ask(question: str, scope: str):
    if not question.strip():
        return "Hãy nhập câu hỏi.", pd.DataFrame()
    try:
        a = ask(question.strip(), index(), docs=SCOPES[scope])
    except RuntimeError as e:  # hết quota LLM
        return f"⚠️ {e}", pd.DataFrame()
    md = f"{BADGE[a.status]}\n\n{a.text}"
    if a.citations:
        md += "\n\n**Nguồn trích dẫn:**\n" + "\n".join(f"- {c}" for c in a.citations)
    return md, pd.DataFrame(a.retrieved, columns=["Đoạn truy xuất", "Độ tương đồng"])


# ---------- Tab 2: yêu cầu ----------
def filter_reqs(dieu: str, ob: str):
    df = REQS
    if dieu != "Tất cả":
        df = df[df.dieu == dieu]
    if ob != "Tất cả":
        df = df[df.obligation == ob]
    show = df.rename(columns={"req_id": "ID", "requirement": "Requirement", "subject": "Subject", "action": "Action",
                              "evidence_control": "Evidence/Control", "obligation": "Mức độ", "source": "Căn cứ"})
    return show[["ID", "Requirement", "Subject", "Action", "Evidence/Control", "Mức độ", "Căn cứ"]]


# ---------- Tab 3: so sánh ----------
def list_changes(status: str, only_summary: bool):
    rows = [c for c in CHANGES if status == "Tất cả" or c["status"] == status]
    return pd.DataFrame([{"#": c["id"], "Điều/Khoản": c["ref"], "Trạng thái": c["status"], "Độ giống": c["similarity"],
                          "Ý nghĩa": c["summary"] or "(chưa có giải thích)", "Thông tư sửa đổi": c["amended_by"]} for c in rows])


def show_change(evt: gr.SelectData):
    c = CHANGES[int(evt.row_value[0])]
    md = f"### {c['ref']} - {c['status']}\n\n**Ý nghĩa:** {c['summary'] or '(chưa có)'}\n\n"
    if c["old"] and c["new"]:
        md += f"**Diff** (~~xóa~~ / **thêm**):\n\n{word_diff(c['old'], c['new'])}\n"
    else:
        md += f"**Nội dung:**\n\n{c['new'] or c['old']}\n"
    imp = c.get("impact") or {}
    if any(imp.values()):
        md += (f"\n**Impact map:** quy trình: {', '.join(imp.get('processes', [])) or '-'} · bộ phận: {', '.join(imp.get('departments', [])) or '-'}"
               f" · trường dữ liệu: {', '.join(imp.get('data_fields', [])) or '-'}")
    return md


# ---------- Tab 4: checklist/rules ----------
def filter_rules(kind: str):
    df = RULES
    if kind == "Kiểm tra tự động được":
        df = df[df.operator != "manual"]
    elif kind == "Kiểm tra thủ công":
        df = df[df.operator == "manual"]
    return df[["req_id", "applies_to", "check", "field", "operator", "value", "severity", "source"]]


# ---------- Tab 5: impact map ----------
def impact_table(by: str):
    rows = []
    for c in CHANGES:
        for v in (c.get("impact") or {}).get(by, []):
            rows.append({by: v, "Điều/Khoản": c["ref"], "Trạng thái": c["status"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(by).agg(**{"Số thay đổi": ("Điều/Khoản", "count"), "Các Điều/Khoản": ("Điều/Khoản", lambda s: ", ".join(sorted(set(s))))})
            .reset_index().sort_values("Số thay đổi", ascending=False))


with gr.Blocks(title="Regulation & Compliance Intelligence") as demo:
    gr.Markdown("# Regulation & Compliance Intelligence Platform\nTra cứu, trích xuất yêu cầu, so sánh phiên bản và sinh checklist từ **Thông tư 39/2016/TT-NHNN** (bản gốc 2016 và bản hợp nhất hiện hành).\n\n" + DISCLAIMER)
    with gr.Tabs():
        with gr.Tab("1. Hỏi đáp có trích dẫn"):
            q = gr.Textbox(label="Câu hỏi", placeholder="VD: Hệ thống cho vay điện tử phải đạt cấp độ an toàn nào?")
            scope = gr.Radio(list(SCOPES), value="Bản hiện hành (VBHN 86)", label="Phạm vi tra cứu")
            btn = gr.Button("Hỏi", variant="primary")
            ans = gr.Markdown()
            ret = gr.Dataframe(label="Các đoạn được truy xuất (để kiểm tra retrieval)", wrap=True)
            gr.Examples(["Tổ chức tín dụng có được cho vay để mua vàng miếng không?",
                         "Hệ thống thông tin cho vay bằng phương tiện điện tử phải đạt cấp độ an toàn nào?",
                         "Cơ cấu lại thời hạn trả nợ phải thực hiện trong bao nhiêu ngày?",
                         "Mức phạt hành chính khi vi phạm quy định cho vay là bao nhiêu?"], inputs=q)
            btn.click(do_ask, [q, scope], [ans, ret])
            q.submit(do_ask, [q, scope], [ans, ret])
        with gr.Tab("2. Yêu cầu trích xuất"):
            gr.Markdown("Mỗi yêu cầu: **Requirement - Subject - Action - Evidence/Control**, kèm căn cứ (Điều/Khoản/trang).")
            with gr.Row():
                f_d = gr.Dropdown(["Tất cả"] + sorted(REQS.dieu.unique(), key=lambda x: (int("".join(filter(str.isdigit, x))), x)), value="Tất cả", label="Điều")
                f_o = gr.Dropdown(["Tất cả", "bắt buộc", "cấm", "cho phép/điều kiện"], value="Tất cả", label="Mức độ")
            t_req = gr.Dataframe(value=filter_reqs("Tất cả", "Tất cả"), wrap=True)
            gr.File(value=str(OUT / "requirements_checklist.xlsx"), label="Tải requirements + checklist (xlsx)")
            for w in (f_d, f_o):
                w.change(filter_reqs, [f_d, f_o], t_req)
        with gr.Tab("3. So sánh phiên bản"):
            gr.Markdown("So sánh **bản gốc 2016** với **bản hợp nhất hiện hành**. Bấm vào một dòng để xem diff chi tiết.")
            st = gr.Radio(["Tất cả", "Added", "Removed", "Modified"], value="Tất cả", label="Lọc theo trạng thái")
            t_ch = gr.Dataframe(value=list_changes("Tất cả", False), wrap=True, interactive=False)
            detail = gr.Markdown()
            gr.File(value=str(OUT / "compare_impact.xlsx"), label="Tải bảng so sánh + impact (xlsx)")
            st.change(lambda s: list_changes(s, False), st, t_ch)
            t_ch.select(show_change, None, detail)
        with gr.Tab("4. Checklist & rules"):
            gr.Markdown("**Checklist** nằm trong file xlsx (sheet Checklist). **Rules** máy đọc được: `manual` = cần người kiểm tra.")
            kind = gr.Radio(["Tất cả", "Kiểm tra tự động được", "Kiểm tra thủ công"], value="Tất cả", label="Loại rule")
            t_rule = gr.Dataframe(value=filter_rules("Tất cả"), wrap=True)
            kind.change(filter_rules, kind, t_rule)
            with gr.Row():
                gr.File(value=str(OUT / "requirements_checklist.xlsx"), label="Checklist (xlsx)")
                gr.File(value=str(OUT / "rules.json"), label="Rules (json)")
        with gr.Tab("5. Impact map (bonus)"):
            gr.Markdown("Quy định thay đổi → quy trình / bộ phận / trường dữ liệu nào bị ảnh hưởng (do LLM đề xuất, cần chuyên gia xác nhận).")
            by = gr.Radio([("Quy trình", "processes"), ("Bộ phận", "departments"), ("Trường dữ liệu", "data_fields")], value="processes", label="Nhóm theo")
            t_imp = gr.Dataframe(value=impact_table("processes"), wrap=True)
            by.change(impact_table, by, t_imp)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
