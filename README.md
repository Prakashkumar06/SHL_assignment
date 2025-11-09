# GenAI Assessment Recommendation Tool

# NOTE (VVI) -> We can make the Recommendation more Accurate but i am using all the FREE resources AND basic models, with limited 0.1cpu usages on RENDER, that why sometime you get wrong suggestion but i try my best. Thank you!

This project recommends the **most relevant assessment** based on a job or hiring requirement.  
It uses **RAG (Retrieval-Augmented Generation)** with **FAISS**, **sentence-transformers**, and **Groq LLM**.

---

## 📌 Features
- Enter any hiring requirement (e.g., "Hiring a Java backend developer").
- System finds the **closest matching assessments** from the product catalog.
- Groq LLM provides a **short explanation** for why the recommendation fits.
- Optionally shows **Top-3 closest matches**.
- Fully **web-based** (no coding needed to use).

## How it will work 
# "- We index **Train-Set** from your Excel using sentence embeddings + FAISS.\n"
# "- For each input, we retrieve Top-5 candidates via FAISS (recall).\n"
# "- If **GROQ_API_KEY** is set, we **re-rank** with Groq (precision).\n"
# "- We show the **best** match, plus optional Top-3 list."







