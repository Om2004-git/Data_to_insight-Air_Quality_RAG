from fastapi import FastAPI
from pydantic import BaseModel
import duckdb
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import pickle

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH = "data/faiss.index"
META_PATH = "data/faiss_meta.pkl"
DB_PATH = "data/gold.duckdb"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"   # make sure this is pulled: ollama pull mistral

# ---------------------------------------

app = FastAPI(title="Air Quality RAG API")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# Connect DuckDB
con = duckdb.connect(DB_PATH, read_only=True)

# Load gold table
gold_df = con.execute("SELECT * FROM air_quality_cleaned").fetchdf()


# ---------------- API Schema ----------------
class QueryRequest(BaseModel):
    q: str


# ---------------- Hybrid Search ----------------
def hybrid_search(query, top_k=3):
    # Vector search
    query_embedding = embed_model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, top_k)

    vector_results = gold_df.iloc[I[0]]

    # SQL keyword search
    sql_results = con.execute("""
        SELECT * FROM air_quality_cleaned
        WHERE city ILIKE '%' || ? || '%'
        LIMIT 3
    """, [query]).fetchdf()

    return vector_results, sql_results


# ---------------- Context Builder ----------------
def build_context(vector_results, sql_results):
    rows = []
    sources = []

    for idx, row in vector_results.iterrows():
        rows.append(
            f"City: {row['city']} | PM2.5: {row['pm25']} | PM10: {row['pm10']} | NO2: {row['no2']} | Date: {row['date']}"
        )
        sources.append(f"row_id: {idx}")

    for idx, row in sql_results.iterrows():
        rows.append(
            f"City: {row['city']} | PM2.5: {row['pm25']} | PM10: {row['pm10']} | NO2: {row['no2']} | Date: {row['date']}"
        )
        sources.append(f"row_id: {idx}")

    context = "\n".join(rows)
    return context, sources


# ---------------- LLM Call ----------------
def call_ollama(context, question):
    prompt = f"""
You are a data analyst assistant.

You MUST answer ONLY using the dataset below.
If the answer is not present, reply exactly:
"Data not available in the dataset."

DATA:
{context}

QUESTION:
{question}

Give a short factual answer.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict data analyst. Never use outside knowledge."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")

    data = response.json()
    return data["message"]["content"]


# ---------------- API Endpoint ----------------
@app.post("/ask")
def ask_question(req: QueryRequest):
    question = req.q

    vector_results, sql_results = hybrid_search(question)

    # Build factual context
    context, sources = build_context(vector_results, sql_results)

    if not context.strip():
        return {
            "answer": "Data not available in the dataset.",
            "sources": ["table: air_quality_cleaned"],
            "confidence": 0.20
        }

    # Call LLM
    answer = call_ollama(context, question)

    return {
        "answer": answer.strip(),
        "sources": ["table: air_quality_cleaned"] + sources,
        "confidence": 0.83
    }


@app.get("/health")
def health():
    return {"status": "ok"}
