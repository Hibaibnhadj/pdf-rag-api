from fastapi import FastAPI
from pydantic import BaseModel
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
import os
import time
nltk.download('punkt')
nltk.download('punkt_tab')
app = FastAPI()
# --- Connexion au modele Granite ---
url = os.environ.get("MODEL_URL")
try:
    with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
        token = f.read().strip()
except FileNotFoundError:
    token = "local-dev-token"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_cache = {}

def get_chunks_and_embeddings(pdf_path):
    mtime = os.path.getmtime(pdf_path)
    cache_key = (pdf_path, mtime)

    if cache_key in embedding_cache:
        print(f"[CACHE] HIT for {pdf_path}", flush=True)
        return embedding_cache[cache_key]

    print(f"[CACHE] MISS for {pdf_path} - computing embeddings", flush=True)
    t0 = time.time()
    text = read_pdf(pdf_path)
    t1 = time.time()
    chunks = split_into_chunks_by_sentence(text)
    t2 = time.time()
    chunk_embeddings = embed_model.encode(chunks)
    t3 = time.time()
    print(f"[TIMING] read_pdf={t1-t0:.2f}s chunk={t2-t1:.2f}s embed={t3-t2:.2f}s", flush=True)

    keys_to_remove = [k for k in embedding_cache if k[0] == pdf_path]
    for k in keys_to_remove:
        del embedding_cache[k]

    embedding_cache[cache_key] = (chunks, chunk_embeddings)
    return chunks, chunk_embeddings

def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def split_into_chunks_by_sentence(text, max_chars=500):
    sentences = sent_tokenize(text, language="french")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def ask_about_pdf_chunked(pdf_path, question, top_k=4):
    chunks, chunk_embeddings = get_chunks_and_embeddings(pdf_path)
    t0 = time.time()
    question_vec = embed_model.encode(question)
    scores = util.cos_sim(question_vec, chunk_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    best_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(best_chunks)
    t1 = time.time()
    prompt = f"Voici des extraits pertinents d'un document :\n\n{context}\n\nReponds a la question suivante en te basant uniquement sur ces extraits :\n{question}"
    payload = {"model": "isvc-granite-31-8b-fp8", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, headers=headers, json=payload, verify=False)
    t2 = time.time()
    print(f"[TIMING] retrieval={t1-t0:.2f}s llm_call={t2-t1:.2f}s", flush=True)
    return response.json()["choices"][0]["message"]["content"]

class Question(BaseModel):
    pdf_path: str
    question: str
@app.post("/ask")
def ask(q: Question):
    answer = ask_about_pdf_chunked(q.pdf_path, q.question)
    return {"answer": answer}
