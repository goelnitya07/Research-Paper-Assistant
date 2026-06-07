import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Research Paper Q&A Assistant",
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

</style>
""", unsafe_allow_html=True)

# =========================
# GEMINI SETUP
# =========================

API_KEY = "YOUR_API_KEY"

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("""
    ### Features

    ✅ Upload Research Papers

    ✅ AI-Powered Question Answering

    ✅ Automatic PDF Text Extraction

    ✅ Research Paper Analysis

    ### Tech Stack

    - Python
    - Streamlit
    - Google Gemini
    - PyPDF

    ### Use Cases

    - Research Assistance
    - Literature Review
    - Academic Study
    - Quick Paper Understanding
    """)

# =========================
# HEADER
# =========================

st.markdown("""
<h1>📚 Research Paper Q&A Assistant</h1>

<p class="subtitle">
Upload a research paper and ask questions using Gemini AI
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

        prompt = f"""
You are an expert research assistant.

Research Paper:
{text[:30000]}

Question:
{question}

Instructions:

- Answer ONLY using information from the paper.
- Use clear section headings.
- Use bullet points wherever appropriate.
- Keep the response structured and easy to read.
- Highlight key findings separately.
- If information is not available in the paper, explicitly state that.
- Format the entire response using Markdown.
"""

        with st.spinner("🤖 Analyzing paper..."):

            try:
                response = model.generate_content(prompt)

                st.markdown("""
                <div class="answer-box">
                <h3>🤖 AI Answer</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(response.text)

            except Exception as e:

                st.error(
                    "Gemini API limit reached. Please wait a minute and try again."
                )

                st.code(str(e))
# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Built using Streamlit + Gemini 2.5 Flash"
)