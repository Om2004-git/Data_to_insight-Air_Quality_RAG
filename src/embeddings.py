import duckdb
import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer



# Config
DB_PATH = "data/gold.duckdb"
FAISS_INDEX_PATH = "data/faiss.index"
META_PATH = "data/faiss_meta.pkl"


# Load Gold Table
print("\nLoading Gold Table from DuckDB...")

con = duckdb.connect(DB_PATH)
tables = con.execute("SHOW TABLES").fetchall()

if not tables:
    raise Exception("No tables found in DuckDB. Run gold_table.py first.")

print("Available Tables:", tables)

df = con.execute("SELECT * FROM air_quality_cleaned").fetchdf()

print("\nGold Dataset Loaded:", df.shape)
print(df.head())



# Create Documents
print("\nCreating documents for embeddings...")

documents = []
metadata = []

for idx, row in df.iterrows():
    text = (
        f"On {row['date']} in {row['city']}, "
        f"PM2.5 was {row['pm25']}, "
        f"PM10 was {row['pm10']}, "
        f"NO2 was {row['no2']}."
    )

    documents.append(text)
    metadata.append({
        "row_id": idx,
        "city": row["city"],
        "date": str(row["date"]),
        "pm25": row["pm25"],
        "pm10": row["pm10"],
        "no2": row["no2"]
    })

print("Total documents:", len(documents))



# Generate Embeddings
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True).astype("float32")



# Create FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)



# Save Index & Metadata
faiss.write_index(index, FAISS_INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("\nFAISS Index Created Successfully!")
print("Saved:", FAISS_INDEX_PATH)
print("Saved:", META_PATH)



# Test Search
def search(query, top_k=3):
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


print("\nTesting semantic search...\n")
query = "What was the pollution in Delhi last month?"
results = search(query)

print("Query:", query)
print("Results:")
for r in results:
    print(r)
