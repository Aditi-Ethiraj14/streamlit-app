import streamlit as st
import pdfplumber
from transformers import pipeline
from fpdf import FPDF
import re
import base64
import docx2txt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from pylatexenc.latex2text import LatexNodes2Text
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="📚 Academic Writing Hub", layout="wide")


# --- Custom CSS for aesthetic glow and styling ---
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f0f0f;
        color: white;
    }

    .reportview-container .main .block-container {
        padding: 2rem 1rem;
        background-color: #1c1c1c;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 255, 150, 0.3);
    }

    h1, h2, h3, h4, h5 {
        color: #90ee90;
    }

    .stButton>button {
        background-color: #222;
        color: white;
        border: 1px solid #90ee90;
        border-radius: 12px;
        padding: 0.5em 1em;
        transition: 0.3s;
    }

    .stButton>button:hover {
        box-shadow: 0 0 10px #90ee90;
        color: black;
        background-color: #90ee90;
    }

    .stTextArea textarea {
        background-color: #2e2e2e !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.75em;
    }

    .stSelectbox > div[role="button"] {
        background-color: #2e2e2e !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.75em;
    }

    .stSelectbox label, .stTextInput label, .stFileUploader label {
        font-weight: bold;
        color: #f1f1f1 !important;
    }
    </style>
""", unsafe_allow_html=True)


# Background music (toggle)
st.sidebar.markdown("**🎵 Music**")
music_toggle = st.sidebar.checkbox("Play background music")
if music_toggle:
    st.markdown("""
    <audio autoplay loop>
      <source src="https://www.bensound.com/bensound-music/bensound-sunny.mp3" type="audio/mpeg">
    </audio>
    """, unsafe_allow_html=True)

# Use lightweight summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
rewriter = pipeline("text2text-generation", model="Falconsai/text_summarization")

# PDF text extraction
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''.join(page.extract_text() or '' for page in pdf.pages)
    return text

# DOCX text extraction
def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

# TEX text extraction
def extract_text_from_tex(uploaded_file):
    tex = uploaded_file.read().decode('utf-8')
    return LatexNodes2Text().latex_to_text(tex)

# Summarize in chunks
def summarize_text(text, max_chunk=1000):
    text = text.replace('\n', ' ')
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ' '.join(summarizer(chunk)[0]['summary_text'] for chunk in chunks)
    return summary.strip()

# Rewrite content

def rewrite_text(text):
    prompt = f"Rewrite the following academic content for clarity and improved grammar:\n{text}"
    return rewriter(prompt)[0]['generated_text']

# Simple metadata extraction
def extract_metadata(text):
    title = text.split('\n')[0][:150]
    authors = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    doi = re.findall(r'doi:\s*\S+|10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text)
    affiliations = re.findall(r'(University|Institute|College|School|Laboratory)[^\n]+', text)
    keywords = re.findall(r'[Kk]eywords:?(.*)', text)
    return {
        "Title": title,
        "Authors": list(set(authors))[:5],
        "Emails": list(set(emails))[:3],
        "DOI": doi[0] if doi else "Not found",
        "Affiliations": list(set(affiliations))[:3],
        "Keywords": keywords[0] if keywords else "Not found"
    }

# Download as .txt
def get_text_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">\U0001F4E5 Download Summary (.txt)</a>'

# Export to PDF
def export_summary_to_pdf(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}\n\nSummary:\n{summary}")
    pdf_output = f"{title[:30].replace(' ', '_')}_summary.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Keyword Cloud
def plot_keyword_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf.getvalue())

# Topic Clusters

def show_topic_clusters(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(topics):
        top_features = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        st.markdown(f"**Topic {topic_idx+1}:** {', '.join(top_features)}")

# Smart Citations via Semantic Scholar

def get_semantic_citations(title):
    try:
        response = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1")
        if response.status_code == 200:
            paper = response.json()['data'][0]
            return f"[{paper['title']}]({paper['url']}) by {', '.join([a['name'] for a in paper['authors']])}"
        else:
            return "Not found."
    except:
        return "Not found."

# --- Streamlit UI ---

st.title("📚 Academic Writing Hub")

mode = st.sidebar.radio("Choose Mode", ["📄 Summarizer", "✍️ Writing Assistant", "⚖️ Domain-Specific Analyzer"])

if mode == "📄 Summarizer":
    st.header("📝 Research Paper Summarizer")
    input_method = st.radio("Input Method", ["📎 Upload PDF", "📄 Upload DOCX", "📃 Upload .TEX", "📝 Paste Text"])
    text = ""

    if input_method == "📎 Upload PDF":
        uploaded_file = st.file_uploader("Upload academic paper (PDF)", type=["pdf"])
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            st.success("✅ PDF extracted.")

    elif input_method == "📄 Upload DOCX":
        docx_file = st.file_uploader("Upload DOCX", type=["docx"])
        if docx_file:
            text = extract_text_from_docx(docx_file)
            st.success("✅ DOCX extracted.")

    elif input_method == "📃 Upload .TEX":
        tex_file = st.file_uploader("Upload LaTeX .tex", type=["tex"])
        if tex_file:
            text = extract_text_from_tex(tex_file)
            st.success("✅ LaTeX extracted.")

    elif input_method == "📝 Paste Text":
        text = st.text_area("Paste text here:", height=300)

    if text:
        if st.button("🚀 Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)
                metadata = extract_metadata(text)
            st.subheader("📌 Summary")
            st.write(summary)

            st.subheader("📄 Metadata")
            for key, value in metadata.items():
                st.markdown(f"**{key}:** {value}")

            st.subheader("🔍 NLP Visualization")
            plot_keyword_cloud(summary)
            show_topic_clusters(summary)

            st.subheader("🔗 Citation Suggestion")
            st.markdown(get_semantic_citations(metadata["Title"]))

            st.markdown(get_text_download_link(summary, "summary.txt"), unsafe_allow_html=True)
            if st.button("📤 Export PDF"):
                file = export_summary_to_pdf(metadata['Title'], summary)
                with open(file, "rb") as f:
                    st.download_button("Download PDF", f, file_name=file)

elif mode == "✍️ Writing Assistant":
    st.header("✍️ AI Writing Assistant")
    format = st.selectbox("Choose Format", ["IEEE", "APA", "MLA", "Chicago"])
    section = st.selectbox("Choose Section", ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"])
    draft = st.text_area("Write or paste your draft:", height=300)

    if st.button("✨ Rewrite for Improvement"):
        with st.spinner("Improving text..."):
            improved = rewrite_text(draft)
        st.subheader("🔁 Suggested Rewrite")
        st.write(improved)

elif mode == "⚖️ Domain-Specific Analyzer":
    st.header("⚖️ Domain-Specific Simplifier")
    domain = st.selectbox("Domain", ["Medical", "Legal"])
    input_method = st.radio("Input Method", ["📎 Upload PDF", "📄 Upload DOCX", "📝 Paste Text"], key=f"{domain}_input")
    domain_text = ""

    if input_method == "📎 Upload PDF":
        domain_file = st.file_uploader(f"Upload {domain} PDF", type=["pdf"], key=f"{domain}_pdf")
        if domain_file:
            domain_text = extract_text_from_pdf(domain_file)
            st.success("✅ PDF Extracted")
    elif input_method == "📄 Upload DOCX":
        domain_file = st.file_uploader(f"Upload {domain} DOCX", type=["docx"], key=f"{domain}_docx")
        if domain_file:
            domain_text = extract_text_from_docx(domain_file)
            st.success("✅ DOCX Extracted")
    else:
        domain_text = st.text_area(f"Paste {domain} text:", height=300, key=f"{domain}_text")

    if domain_text:
        if st.button(f"🧠 Summarize {domain} Content"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(domain_text)
            st.subheader(f"Summary for {domain} Document")
            st.write(summary)
            plot_keyword_cloud(summary)
            show_topic_clusters(summary)
