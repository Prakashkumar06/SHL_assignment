# import faiss
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os

# # all-Paths
# INDEX_PATH = "data/catalog.index"
# DF_PATH = "data/catalog.pkl"
# MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# # Lazy singletons
# _df = None
# _index = None
# _model = None

# def _load_df():
#     global _df
#     if _df is None:
#         _df = pd.read_pickle(DF_PATH)
#     return _df

# def _load_index():
#     global _index
#     if _index is None:
#         _index = faiss.read_index(INDEX_PATH)
#     return _index

# def _load_model():
#     global _model
#     if _model is None:
#         _model = SentenceTransformer(MODEL_NAME)
#     return _model

# def retrieve_candidates(user_query: str, k: int = 5) -> pd.DataFrame:
#     """Return top-k candidates with FAISS distances ."""
#     df = _load_df()
#     index = _load_index()
#     model = _load_model()

#     emb = model.encode([user_query], convert_to_numpy=True).astype(np.float32)
#     dists, idxs = index.search(emb, k)
#     out = df.iloc[idxs[0]].copy()
#     out.insert(0, "faiss_distance", dists[0])
#     return out.reset_index(drop=True)

# def top1_faiss(user_query: str) -> pd.Series:
#     """Simple baseline: FAISS top-1 --no LLM."""
#     return retrieve_candidates(user_query, k=1).iloc[0]

# New code we use light-weght model to reduce CPU load 

import os
import pickle
import faiss
import numpy as np
import pandas as pd


BASE = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE, "..", "data", "catalog.index")
META_PATH  = os.path.join(BASE, "..", "data", "catalog.pkl")

with open(META_PATH, "rb") as f:
    pack = pickle.load(f)
df = pack["df"]
X  = pack["X"]
vectorizer = pack["vectorizer"]

faiss_index = faiss.read_index(INDEX_PATH)

def embed_query(q: str) -> np.ndarray:
    return vectorizer.transform([q]).astype("float32").toarray()

def retrieve_candidates(query: str, k: int = 5) -> pd.DataFrame:
    qv = embed_query(query)
    dists, idxs = faiss_index.search(qv, k)
    out = df.iloc[idxs[0]].copy()
    out.insert(0, "faiss_distance", dists[0])
    return out.reset_index(drop=True)

def top1_faiss(query: str):
    return retrieve_candidates(query, 1).iloc[0].to_dict()
