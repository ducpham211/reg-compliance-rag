# Regulation & Compliance Intelligence Platform (Project 06)

Demo RAG + trích xuất yêu cầu + so sánh phiên bản trên **Thông tư 39/2016/TT-NHNN** (bản gốc 2016 vs. văn bản hợp nhất 86/VBHN-NHNN hiện hành).
Bản demo học tập, không phải tư vấn pháp lý.

## Chạy
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# tạo file .env gồm 1 dòng GEMINI_API_KEY=<key thật> (key miễn phí từ aistudio.google.com), KHÔNG để chữ '...'
PYTHONPATH=src python app.py                # mở http://127.0.0.1:7860
```
Chỉ cần `poppler-utils` và `tesseract-ocr` + `tesseract-ocr-vie` khi OCR lại bản gốc (file scan). Kết quả OCR đã lưu ở `docs/tt39_2016_goc/pages/`.

## Build lại dữ liệu (offline, kết quả LLM được cache)
```bash
PYTHONPATH=src python scripts/build.py [parse|compare|extract|rules|export|all]
PYTHONPATH=src python scripts/eval.py      # -> data/output/eval_report.md
```

## Cấu trúc
- `src/regrag/parse.py` tách Điều/Khoản + số trang + thông tư sửa đổi (lấy từ chân trang VBHN)
- `retrieve.py` hybrid embedding + BM25 | `qa.py` hỏi đáp, citation lấy từ metadata, từ chối khi thiếu căn cứ
- `compare.py` Added/Removed/Modified (thuật toán, không dùng LLM) | `analyze.py` giải thích, impact map, yêu cầu, rules
- `llm.py` Gemini free tier (xoay vòng model, cache đĩa) hoặc Ollama (`LLM_PROVIDER=ollama`)
- `eval/golden.json` bộ câu hỏi vàng (24 có đáp án + 6 không có căn cứ)

## Lưu ý
- Free tier của Gemini giới hạn theo model và theo ngày (có model chỉ 20 request/ngày); `llm.py` tự chuyển model và dùng cache.
- Giá trị `gradio==0.50.0` trong ghi chú gốc không tồn tại; dự án dùng `gradio==5.50.0`.
