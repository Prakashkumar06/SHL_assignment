# import pandas as pd
# import faiss
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os

# DATA_XLSX = os.getenv("PRODUCT_XLSX_PATH", "data/Gen_AI Dataset.xlsx")
# OUT_INDEX = "data/catalog.index"
# OUT_DF_PKL = "data/catalog.pkl"
# MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# def main():
#     df = pd.read_excel(DATA_XLSX, sheet_name="Train-Set")
#     if "Query" not in df.columns or "Assessment_url" not in df.columns:
#         raise ValueError("Train-Set must contain 'Query' and 'Assessment_url' columns.")

#     model = SentenceTransformer(MODEL_NAME)
#     embeddings = model.encode(df["Query"].tolist(), convert_to_numpy=True).astype("float32")

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     faiss.write_index(index, OUT_INDEX)
#     df.to_pickle(OUT_DF_PKL)

#     print("Train-Set indexed succesdfully")
#     print(f"Saved index: {OUT_INDEX}")
#     print(f"Saved DataFrame: {OUT_DF_PKL}")

# if __name__ == "__main__":
#     main()


#   (TF-IDF version)
import pandas as pd
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DATA_XLSX = "data/Gen_AI Dataset.xlsx"
OUT_INDEX = "data/catalog.index"
OUT_PKL = "data/catalog.pkl"

df = pd.read_excel(DATA_XLSX, sheet_name="Train-Set")
df = df.dropna(subset=["Query","Assessment_url"]).reset_index(drop=True)


vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X = vectorizer.fit_transform(df["Query"]).astype("float32").toarray()

index = faiss.IndexFlatL2(X.shape[1])
index.add(X)


with open(OUT_PKL, "wb") as f:
    pickle.dump({"df": df, "X": X, "vectorizer": vectorizer}, f)


faiss.write_index(index, OUT_INDEX)
print("TF-IDF build:", OUT_INDEX, "| meta:", OUT_PKL)
