import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

DATA_XLSX = os.getenv("PRODUCT_XLSX_PATH", "data/Gen_AI Dataset.xlsx")
OUT_INDEX = "data/catalog.index"
OUT_DF_PKL = "data/catalog.pkl"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def main():
    # Load the trein-Set only
    df = pd.read_excel(DATA_XLSX, sheet_name="Train-Set")
    if "Query" not in df.columns or "Assessment_url" not in df.columns:
        raise ValueError("Train-Set must contain 'Query' and 'Assessment_url' columns.")

    # Build the embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["Query"].tolist(), convert_to_numpy=True)

    # Create FAISS (L2) index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    # Save index + dataframe
    faiss.write_index(index, OUT_INDEX)
    df.to_pickle(OUT_DF_PKL)

    print(" Train-Set indexed succesdfully")
    print(f" Saved index: {OUT_INDEX}")
    print(f" Saved data : {OUT_DF_PKL}")

if __name__ == "__main__":
    main()
