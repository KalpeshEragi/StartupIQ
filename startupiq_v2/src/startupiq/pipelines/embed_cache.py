"""Encode every usable YC description with a pretrained sentence encoder and cache it (chunked, resumable).
Run: PYTHONPATH=src python -m startupiq.pipelines.embed_cache MODEL [MAXLEN]   (repeat until it prints DONE)
Texts are encoded once; no label or split information is used, so caching cannot leak."""
import os, sys, time
import numpy as np
from startupiq.pipelines.yc_split import load_split

PREFIX = {"intfloat/e5-small-v2": "passage: ", "intfloat/e5-base-v2": "passage: "}
DIR = "data/interim/emb"


def path(model, maxlen, col="text"): return f"{DIR}/{model.replace('/', '_')}_{maxlen}{'' if col == 'text' else '_' + col}.npy"


def load_embeddings(model, maxlen=256):
    df, *_ = load_split()
    E = np.load(path(model, maxlen)); return df, E


if __name__ == "__main__":
    model = sys.argv[1]; maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 256; col = sys.argv[3] if len(sys.argv) > 3 else "text"
    os.makedirs(DIR, exist_ok=True); T0 = time.time()
    df, *_ = load_split(); texts = [PREFIX.get(model, "") + str(t) for t in df[col].fillna("").tolist()]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device="cpu"); m.max_seq_length = maxlen
    final = path(model, maxlen, col); part = final + ".part.npy"
    done = np.load(part) if os.path.exists(part) else np.zeros((0, m.get_sentence_embedding_dimension()), dtype=np.float32)
    B = 256
    while len(done) < len(texts):
        if time.time() - T0 > 110: np.save(part, done); print("progress", len(done), "/", len(texts)); raise SystemExit("rerun")
        chunk = texts[len(done):len(done) + B]
        e = m.encode(chunk, batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        done = np.vstack([done, e])
    np.save(final, done); print("DONE", model, done.shape, round(time.time() - T0, 1), "s")
