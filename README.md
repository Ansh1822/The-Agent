# Ledger & Lens — RAG-Grounded Data & Document Analyst

A full-stack Python AI Data Analyst workbench built with Streamlit, Groq (`openai/gpt-oss-120b`), FAISS, FastEmbed, and Plotly.

## Features
- **Universal Non-Media Ingestion:** Supports CSV, TSV, Excel (`.xlsx`, `.xls`), JSON, Parquet, PDF, DOCX, TXT, Markdown, and log files.
- **Hybrid RAG + Python Execution:** Combines semantic vector retrieval (FAISS + BGE-Small) with deterministic Pandas and Plotly code execution.
- **Live Chart Interrogation:** Ask follow-up questions directly about the active chart while inspecting it side-by-side.
- **Editorial Print UI:** Tactile paper-and-ink design (`#F7F5F0` alabaster, `#18181B` ink, `#C84B31` terracotta accents).

## Local Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py