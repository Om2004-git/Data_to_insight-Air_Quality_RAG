Air Quality Data Platform — RAG System
Overview

This project implements an end-to-end Data Engineering + AI RAG pipeline for air quality analytics.

It includes:

Spark ETL pipeline

Gold KPI layer (DuckDB)

FAISS vector database

Hybrid SQL + Vector retrieval

FastAPI RAG service

Local LLM via Ollama

Architecture
           ┌──────────────┐
           │   Raw CSV    │
           └──────┬───────┘
                  │
          Spark ETL Pipeline
                  │
         ┌────────▼─────────┐
         │  Cleaned Delta   │
         └────────┬─────────┘
                  │
           Gold Table (DuckDB)
                  │
       ┌──────────▼──────────┐
       │ KPI Tables + Facts  │
       └──────────┬──────────┘
                  │
         FAISS Vector Index
                  │
         Hybrid Retrieval Layer
           (SQL + FAISS)
                  │
              FastAPI
              /ask API
                  │
              Ollama LLM

Tech Stack:
Layer	Technology
ETL	PySpark
Storage	Parquet
Gold Layer	DuckDB
Vector DB	FAISS
Embeddings	Sentence Transformers
API	FastAPI
LLM	Ollama (Mistral)

Setup Instructions
1. Create Environment
python -m venv data_to_insight
data_to_insight\Scripts\activate
pip install -r requirements.txt

2️. Run Spark Pipeline
python src/spark_pipeline.py


Creates cleaned Parquet files.

3️. Create Gold Tables
python src/gold_table.py


Creates:

air_quality_cleaned

kpi_city

kpi_monthly

kpi_top_polluted

4. Build FAISS Index
python src/embeddings.py


Creates:

data/faiss.index

data/faiss_meta.pkl

5️. Start Ollama
ollama run mistral

6️. Start API Server
uvicorn src.app:app --reload


Open:

http://127.0.0.1:8000/docs

API Usage
POST /ask
{
  "q": "What is the average PM2.5 in Delhi?"
}


Returns:

{
  "answer": "...",
  "sources": ["table: air_quality_cleaned", "row_id: ..."],
  "confidence": 0.83
}

Example Queries

"What is the average PM2.5 in Delhi?"

"Which is the most polluted city?"

"Show monthly pollution trend"

"Which city has highest NO2?"

