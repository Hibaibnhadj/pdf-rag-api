from fastapi import FastAPI
from pydantic import BaseModel
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
import os

nltk.download('punkt')
nltk.download('punkt_tab')

app = FastAPI()

# --- Connexion au modèle Granite ---
url = os.environ.get("MODEL_URL")

with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
    token = f.read().strip()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# --- Modèle d'embeddings (chargé une seule fois au démarrage) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Lecture de PDF ---
def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# --- Chunking basé sur les phrases ---
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

# --- Retrieval + réponse ---
def ask_about_pdf_chunked(pdf_path, question, top_k=4):
    text = read_pdf(pdf_path)
    chunks = split_into_chunks_by_sentence(text)
    chunk_embeddings = embed_model.encode(chunks)

    question_vec = embed_model.encode(question)
    scores = util.cos_sim(question_vec, chunk_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    best_chunks = [chunks[i] for i in top_indices]

    context = "\n\n".join(best_chunks)
    prompt = f"Voici des extraits pertinents d'un document :\n\n{context}\n\nRéponds à la question suivante en te basant uniquement sur ces extraits :\n{question}"
    payload = {"model": "isvc-granite-31-8b-fp8", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, headers=headers, json=payload, verify=False)
    return response.json()["choices"][0]["message"]["content"]

# --- Endpoint ---
class Question(BaseModel):
    pdf_path: str
    question: str

@app.post("/ask")
def ask(q: Question):
    answer = ask_about_pdf_chunked(q.pdf_path, q.question)
    return {"answer": answer}