"""RAG knowledge base: load -> chunk -> hybrid index (BM25 keyword + dense embeddings) -> RRF fusion.

* BM25 is pure Python, so retrieval works even without any ML libraries (fast, offline, testable).
* Dense retrieval (HuggingFace MiniLM embeddings + Chroma) is switched on automatically when the
  packages are installed; results from both are merged with Reciprocal Rank Fusion (RRF).
"""
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config

STOPWORDS = set("""a an and are as at be by for from has have how i in is it its of on or that the this to was what
when where which who why will with you your do does did can could should about into than then them they we""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None


# ---------------------------------------------------------------- loading ----
def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return [(text, page_number_or_None)] for pdf / docx / txt / md files."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [((pg.extract_text() or ""), i + 1) for i, pg in enumerate(reader.pages)]
    if ext == ".docx":
        import docx
        d = docx.Document(str(path))
        return [("\n\n".join(p.text for p in d.paragraphs if p.text.strip()), None)]
    if ext in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    raise ValueError(f"Unsupported file type: {ext} (use pdf, docx, txt or md)")


def _hard_split(paragraph: str, size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, buf = [], ""
    for s in sentences:
        while len(s) > size:                     # a single huge sentence
            pieces.append(s[:size]); s = s[size:]
        if len(buf) + len(s) + 1 <= size:
            buf = f"{buf} {s}".strip()
        else:
            pieces.append(buf); buf = s
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p]


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking with a small overlap so ideas aren't cut in half."""
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        for piece in ([para] if len(para) <= size else _hard_split(para, size)):
            if len(buf) + len(piece) + 2 <= size:
                buf = f"{buf}\n\n{piece}" if buf else piece
            else:
                if buf:
                    chunks.append(buf)
                tail = buf[-overlap:] if (buf and overlap) else ""
                tail = tail[tail.find(" ") + 1:] if " " in tail else ""   # start on a word boundary
                buf = f"{tail} {piece}".strip() if tail else piece
    if buf:
        chunks.append(buf)
    return chunks


# ------------------------------------------------------------------- BM25 ----
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.N) if self.N else 0.0
        self.df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.tf = [self._count(d) for d in docs]

    @staticmethod
    def _count(doc):
        c: dict[str, int] = {}
        for t in doc:
            c[t] = c.get(t, 0) + 1
        return c

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.N
        for t in set(query):
            n = self.df.get(t, 0)
            if not n:
                continue
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    dl = len(self.docs[i])
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


# ---------------------------------------------------------- knowledge base ----
class KnowledgeBase:
    def __init__(self, index_dir=None, use_dense: bool | None = None):
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.index_dir / "chunks.json"
        self.chunks: list[Chunk] = []
        self._load()
        self._rebuild_bm25()
        self._chroma = None
        self.dense_enabled = False
        if config.USE_DENSE if use_dense is None else use_dense:
            self._init_dense()

    # persistence
    def _load(self):
        if self._file.exists():
            try:
                self.chunks = [Chunk(**c) for c in json.loads(self._file.read_text(encoding="utf-8"))]
            except (json.JSONDecodeError, TypeError, OSError):
                self.chunks = []

    def _save(self):
        self._file.write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8")

    def _rebuild_bm25(self):
        self._id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
        self._bm25 = BM25([tokenize(c.text + " " + c.source) for c in self.chunks])

    # dense index (optional)
    def _init_dense(self):
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
            self._chroma = Chroma(collection_name="academic_notes", embedding_function=emb,
                                  persist_directory=str(self.index_dir / "chroma"),
                                  collection_metadata={"hnsw:space": "cosine"})
            self.dense_enabled = True
            self._sync_dense()
        except Exception as exc:  # missing packages / offline -> BM25 only
            print(f"[rag] dense retrieval disabled ({type(exc).__name__}: {exc}); using BM25 only")
            self._chroma, self.dense_enabled = None, False

    def _sync_dense(self):
        try:
            existing = set(self._chroma.get(include=[])["ids"])
        except Exception:
            existing = set()
        self._add_dense([c for c in self.chunks if c.id not in existing])

    def _add_dense(self, chunks: list[Chunk]):
        if self._chroma is not None and chunks:
            self._chroma.add_texts([c.text for c in chunks], ids=[c.id for c in chunks],
                                   metadatas=[{"id": c.id, "source": c.source, "page": c.page or 0} for c in chunks])

    # ingestion
    def add_text(self, text: str, source: str, page: int | None = None) -> int:
        known = {c.id for c in self.chunks}
        new = []
        for t in chunk_text(text):
            cid = hashlib.sha1(f"{source}|{page}|{t}".encode()).hexdigest()[:16]
            if cid not in known:
                new.append(Chunk(cid, t, source, page))
                known.add(cid)
        if new:
            self.chunks.extend(new)
            self._save()
            self._rebuild_bm25()
            self._add_dense(new)
        return len(new)

    def add_file(self, path) -> int:
        path = Path(path)
        return sum(self.add_text(t, path.name, pg) for t, pg in load_file(path))

    def add_folder(self, folder) -> int:
        total = 0
        for p in sorted(Path(folder).glob("*")):
            if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
                total += self.add_file(p)
        return total

    def clear(self):
        self.chunks = []
        self._save()
        self._rebuild_bm25()
        if self._chroma is not None:
            try:
                self._chroma.delete_collection()
            except Exception:
                pass
            self._init_dense()

    # query
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def chunks_of(self, source: str) -> list[Chunk]:
        return [c for c in self.chunks if c.source.lower() == source.lower()]

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Hybrid search. Returns [{text, source, page, score}] best first."""
        if not self.chunks:
            return []
        pool = k * 3
        bm = self._bm25.scores(tokenize(query))
        bm_rank = [i for i in sorted(range(len(bm)), key=lambda i: -bm[i]) if bm[i] > 0][:pool]
        dense_rank: list[int] = []
        if self.dense_enabled:
            try:
                for doc, dist in self._chroma.similarity_search_with_score(query, k=pool):
                    idx = self._id_to_idx.get(doc.metadata.get("id"))
                    if idx is not None and dist <= config.DENSE_MAX_DISTANCE:
                        dense_rank.append(idx)
            except Exception as exc:
                print(f"[rag] dense search failed: {exc}")
        fused: dict[int, float] = {}
        for ranking in (bm_rank, dense_rank):          # Reciprocal Rank Fusion
            for r, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + r + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{"text": self.chunks[i].text, "source": self.chunks[i].source,
                 "page": self.chunks[i].page, "score": round(fused[i], 4)} for i in top]
