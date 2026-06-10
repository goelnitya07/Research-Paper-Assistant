import streamlit as st
from pypdf import PdfReader
import ollama

import faiss
import numpy as np

@st.cache_resource
def create_vector_store(text):

    chunk_size = 300

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(
            text[i:i + chunk_size]
        )

    embedding_response = ollama.embed(
        model="nomic-embed-text",
        input=chunks
    )

    embeddings = np.array(
        embedding_response["embeddings"]
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return chunks, index

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #4CAF50;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.answer-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #4CAF50;
    margin-top: 20px;
}

.sidebar .sidebar-content {
    background-color: #111111;
}

.stButton>button {
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
}

.stTextInput input {
    border-radius: 10px;
}

[data-testid="stMetricValue"] {
    color: #4CAF50;
}

hr {
    border: 1px solid #222;
}

</style>
""", unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("""
    ### Features

✅ Upload Research Papers

✅ Semantic Search with FAISS

✅ Local LLM Question Answering

✅ Automatic PDF Text Extraction

✅ Retrieval-Augmented Generation (RAG)

### Tech Stack

- Python
- Streamlit
- Ollama
- Qwen3
- FAISS
- nomic-embed-text
- NumPy
- PyPDF

### Use Cases

- Research Assistance
- Literature Review
- Academic Study
- Quick Paper Understanding
- Semantic Document Search
    """)

# =========================
# HEADER
# =========================

st.markdown("""
<h1>📚 Research Paper RAG Assistant</h1>

<p class="subtitle">
Upload a research paper and ask questions using Local RAG Pipeline powered by Qwen3
</p>
""", unsafe_allow_html=True)

# =========================
# FILE UPLOAD
# =========================

pdf = st.file_uploader(
    "Upload a Research Paper (PDF)",
    type="pdf"
)

# =========================
# MAIN APP
# =========================

if pdf:

    with st.spinner("Reading PDF..."):

        reader = PdfReader(pdf)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        chunks, index = create_vector_store(text) 

    st.success("✅ PDF Loaded Successfully!")

    # Metrics

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pages",
            len(reader.pages)
        )

    with col2:
        st.metric(
            "Characters Extracted",
            len(text)
        )

    st.info("""
### Example Questions

• What is the main contribution of this paper?

• Summarize the methodology.

• What datasets were used?

• What are the key findings?

• What future work is suggested?

• What limitations are discussed?
""")

    question = st.text_input(
        "🔍 Ask a question about the paper"
    )

    if question:

        question_embedding = ollama.embed(
            model="nomic-embed-text",
            input=question
        )

        question_vector = np.array(
            [question_embedding["embeddings"][0]]
        ).astype("float32")

        D, I = index.search(
            question_vector,
            k=2
        )

        retrieved_chunks = []

        for idx in I[0]:
            retrieved_chunks.append(
                chunks[idx]
            )

        context = "\n\n".join(
            retrieved_chunks
        )

        prompt = f"""
You are an expert research assistant.

Context from the paper:

{context}

Question:

{question}

Instructions:

- Answer ONLY using information from the context.
- Use headings and bullet points.
- Be concise and well structured.
- If the answer is not present, say so.
"""

        with st.spinner("🤖 Analyzing paper..."):

            try:

                response = ollama.chat(
                    model="qwen3:1.7b",
                    messages=[
                        {
                            "role":"system",
                            "content":"Answer directly. Do not reveal reasoning. Do not think aloud."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": 300
                    }
                )

                answer = response["message"]["content"]

                st.markdown("""
                <div class="answer-box">
                <h3>🤖 AI Answer</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(answer)

            except Exception as e:

                st.error(
                    "Could not connect to Ollama."
                )

                st.code(str(e))
# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Built using Streamlit + Ollama + Qwen3 + FAISS"
)