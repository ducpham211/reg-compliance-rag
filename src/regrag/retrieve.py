"""Truy xuất hybrid (embedding + BM25, hợp nhất bằng RRF) trên các chunk Điều/Khoản."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .parse import Clause, load

DOC_LABEL = {
    "vbhn_86": "TT 39/2016/TT-NHNN (VBHN 86, hiện hành)",
    "tt39_2016_goc": "TT 39/2016/TT-NHNN (bản gốc 2016)",
}
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CACHE = Path("data/cache")


def fold(s: str) -> str:
    """Bỏ dấu + hạ chữ thường: chống lỗi dấu của OCR và người dùng gõ không dấu."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def tokenize(s: str) -> list[str]:
    w = re.findall(r"\w+", fold(s))
    return w + [a + "_" + b for a, b in zip(w, w[1:])]  # bigram âm tiết ~ từ ghép tiếng Việt


def citation(c: Clause) -> str:
    return f"{DOC_LABEL.get(c.doc_id, c.doc_id)}, {c.ref}, tr.{c.page}"


def chunk_text(c: Clause) -> str:
    return f"{c.ref} ({c.dieu_title}): {c.text}"


class Index:
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses
        self.texts = [chunk_text(c) for c in clauses]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.emb = self._embed_corpus()

    @classmethod
    def from_processed(cls, names=("vbhn_86", "tt39_2016_goc")) -> "Index":
        cl: list[Clause] = []
        for n in names:
            cl += load(f"data/processed/{n}.json")
        return cls(cl)

    @staticmethod
    def _model():
        from fastembed import TextEmbedding
        return TextEmbedding(EMBED_MODEL)

    def _embed_corpus(self) -> np.ndarray:
        tag = re.sub(r"\W+", "_", EMBED_MODEL) + f"_{len(self.texts)}_{sum(map(len, self.texts))}"
        f = CACHE / f"emb_{tag}.npy"
        if f.exists():
            return np.load(f)
        self._m = self._model()
        e = np.array(list(self._m.embed(self.texts)), dtype="float32")
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, e)
        return e

    def search(self, q: str, k: int = 6, docs: tuple[str, ...] | None = None):
        """Trả list (Clause, điểm_cosine, hạng_hợp_nhất). Điểm cosine dùng để ước lượng 'có căn cứ hay không'."""
        if not hasattr(self, "_m"):
            self._m = self._model()
        qv = np.array(list(self._m.embed([q])), dtype="float32")[0]
        qv /= np.linalg.norm(qv)
        cos = self.emb @ qv
        bm = self.bm25.get_scores(tokenize(q))
        rrf = np.zeros(len(self.clauses))
        for order in (np.argsort(-cos), np.argsort(-bm)):
            for rank, i in enumerate(order):
                rrf[i] += 1.0 / (60 + rank)
        out = []
        for i in np.argsort(-rrf):
            if docs and self.clauses[i].doc_id not in docs:
                continue
            out.append((self.clauses[i], float(cos[i]), float(bm[i])))
            if len(out) == k:
                break
        return out
