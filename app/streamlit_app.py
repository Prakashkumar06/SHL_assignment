import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from src.recommender import retrieve_candidates, top1_faiss
from src.groq_helper import groq_available, rerank_candidates, explain_recommendation

st.set_page_config(page_title="GenAI Assessment Recommender", layout="centered")

st.title("GenAI Assessment Recommendation Tool")


query = st.text_input("Describe the role or hiring requirement:")
k_recall = st.slider("FAISS recall (k)", 3, 10, 5)
show_top3 = st.checkbox("Also show Top-3 Suggestion", value=True)

if st.button("Recommend", type="primary"):
    if not query.strip():
        st.warning("Please enter a requirement first.")
        st.stop()

    with st.spinner("Retrieving candidates..."):
        cands = retrieve_candidates(query, k=k_recall)

    if groq_available():
        with st.spinner("Re-ranking with Groq..."):
            best_idx_1based = rerank_candidates(query, cands)
    else:
        st.info("GROQ_API_KEY not set — using FAISS top-1 without re-rank.")
        best_idx_1based = 1

    best = cands.iloc[best_idx_1based - 1]

    st.subheader("Best Recommended Assessment")
    st.write(f"**Match (catalog descriptor):** {best['Query']}")
    st.markdown(f"[Open Assessment]({best['Assessment_url']})")
    st.caption(f"FAISS distance: {best['faiss_distance']:.4f} (lower is better)")

    with st.spinner("Generating explanation..." if groq_available() else "Explanation unavailable"):
        explanation = explain_recommendation(query, best["Query"])
    st.markdown("### Why this recommendation?")
    st.write(explanation)

    if show_top3:
        st.markdown("---")
        st.subheader("Top-3 Sugesstions(you can check this also-->>)")
        top3 = cands.head(3).copy()
        # Emphasize the winner
        for i, row in top3.reset_index(drop=True).iterrows():
            bullet = "TOP" if (i + 1) == best_idx_1based else "•"
            st.markdown(
                f"{bullet} **{row['Query']}**  \n"
                f"FAISS distance: `{row['faiss_distance']:.4f}`  \n"
                f"[Link]({row['Assessment_url']})"
            )

st.markdown("---")
st.caption("Make sure to run `python src/build_index.py` before using the tool.")
