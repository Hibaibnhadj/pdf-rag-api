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
# --- Modele d'embeddings (charge une seule fois au demarrage) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Cache en memoire des embeddings deja calcules ---
# Cle: (chemin du PDF, date de derniere modification du fichier)
# Valeur: (liste des chunks, embeddings correspondants)
# Ce cache evite de relire/redecouper/reembedder le meme PDF a chaque question.
# Limite connue: le cache vit en RAM du pod -> il est perdu si le pod redemarre,
# et il ne resout pas la consommation memoire lors de requetes concurrentes sur
# des PDFs DIFFERENTS (chacun garde sa propre entree de cache en memoire).
embedding_cache = {}

def get_chunks_and_embeddings(pdf_path):
    mtime = os.path.getmtime(pdf_path)
    cache_key = (pdf_path, mtime)

    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    text = read_pdf(pdf_path)
    chunks = split_into_chunks_by_sentence(text)
    chunk_embeddings = embed_model.encode(chunks)

    keys_to_remove = [k for k in embedding_cache if k[0] == pdf_path]
    for k in keys_to_remove:
        del embedding_cache[k]

    embedding_cache[cache_key] = (chunks, chunk_embeddings)
    return chunks, chunk_embeddings

# --- Lecture de PDF ---
def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
# --- Chunking base sur les phrases ---
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
# --- Retrieval + reponse ---
def ask_about_pdf_chunked(pdf_path, question, top_k=4):
    chunks, chunk_embeddings = get_chunks_and_embeddings(pdf_path)
    question_vec = embed_model.encode(question)
    scores = util.cos_sim(question_vec, chunk_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    best_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(best_chunks)
    prompt = f"Voici des extraits pertinents d'un document :\n\n{context}\n\nReponds a la question suivante en te basant uniquement sur ces extraits :\n{question}"
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
