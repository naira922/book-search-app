import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -------------------------------
# 1. Preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = simple_preprocess(text, deacc=True)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

# -------------------------------
# 2. Load PDF & Split
# -------------------------------
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return [page.get_text().strip() for page in doc if page.get_text().strip()]

def split_by_chapter(pages):
    chapters = {}
    chapter_idx = 0
    current_text = []
    for i, page in enumerate(pages):
        if re.match(r"chapter\s+\d+", page.lower()):
            if current_text:
                chapters[f"Chapter {chapter_idx}"] = " ".join(current_text)
            chapter_idx += 1
            current_text = [page]
        else:
            current_text.append(page)
    if current_text:
        chapters[f"Chapter {chapter_idx}"] = " ".join(current_text)
    return chapters

# -------------------------------
# 3. Train Word2Vec + Vectorize
# -------------------------------
def build_word2vec(chunks):
    tokenized = [preprocess(text) for text in chunks]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=2)
    return model

def vectorize_text(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# -------------------------------
# 4. Search
# -------------------------------
def search(query, chunks, model):
    qv = vectorize_text(preprocess(query), model).reshape(1, -1)
    cvs = np.array([vectorize_text(preprocess(c), model) for c in chunks])
    sims = cosine_similarity(qv, cvs)[0]
    return sorted(zip(range(len(chunks)), chunks, sims), key=lambda x: x[2], reverse=True)[:5]


st.set_page_config(page_title="Smart Book Search", page_icon="📚")
st.title("📚 Smart Book Search (Word2Vec)")

uploaded = st.file_uploader("Upload a PDF book", type="pdf")
if uploaded:
    pages = load_pdf(uploaded)

    
    option = st.radio("Choose search mode:", ["By Page", "By Chapter", "Whole Book"])

    if option == "By Page":
        chunks = pages
    elif option == "By Chapter":
        chapters = split_by_chapter(pages)
        chunks = list(chapters.values())
    else:
        chunks = [" ".join(pages)]  # full book as one chunk

    st.write(f"✅ Loaded {len(chunks)} chunks")

    
    model = build_word2vec(chunks)

    
    query = st.text_input("🔍 Enter your search query:")
    if query:
        results = search(query, chunks, model)
        st.markdown("### 🔎 Top Results:")
        for rank, (idx, text, score) in enumerate(results, start=1):
            st.markdown(f"**{rank}. Chunk {idx + 1} (Score: {score:.4f})**")
            with st.expander("📖 Show full text"):
                st.text_area("Full snippet", value=text, height=200)



