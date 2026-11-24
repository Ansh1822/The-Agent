import io
import json
import pandas as pd
from pypdf import PdfReader
from docx import Document


TABULAR_EXTS = {"csv", "tsv", "xlsx", "xls", "json", "parquet"}
TEXT_EXTS = {"pdf", "docx", "txt", "md", "log"}


def parse_uploaded_file(uploaded_file) -> dict:
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    raw_bytes = uploaded_file.read()
    buffer = io.BytesIO(raw_bytes)

    result = {
        "filename": filename,
        "ext": ext,
        "is_tabular": False,
        "df": None,
        "text_blocks": [],
        "summary": "",
    }

    if ext in TABULAR_EXTS:
        df = _load_tabular(buffer, ext)
        result["is_tabular"] = True
        result["df"] = df
        result["text_blocks"] = _dataframe_to_rag_documents(df, filename)
        result["summary"] = (
            f"Tabular Dataset ({df.shape[0]:,} rows × {df.shape[1]} columns)"
        )
    elif ext in TEXT_EXTS:
        text = _load_text(buffer, ext)
        result["text_blocks"] = _chunk_text(text, filename)
        result["summary"] = (
            f"Document ({len(text):,} characters, {len(result['text_blocks'])} indexed chunks)"
        )
    else:
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
            if not text.strip():
                raise ValueError("File appears to be empty or binary.")
            result["text_blocks"] = _chunk_text(text, filename)
            result["summary"] = f"Text File ({len(result['text_blocks'])} indexed chunks)"
        except Exception as e:
            raise ValueError(f"Unsupported file format '.{ext}': {e}")

    return result


def _load_tabular(buffer: io.BytesIO, ext: str) -> pd.DataFrame:
    if ext == "csv":
        return pd.read_csv(buffer)
    if ext == "tsv":
        return pd.read_csv(buffer, sep="\t")
    if ext in {"xlsx", "xls"}:
        return pd.read_excel(buffer)
    if ext == "parquet":
        return pd.read_parquet(buffer)
    if ext == "json":
        data = json.load(buffer)
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return pd.json_normalize(val)
            return pd.json_normalize([data])
    raise ValueError(f"Could not parse tabular file: {ext}")


def _load_text(buffer: io.BytesIO, ext: str) -> str:
    if ext == "pdf":
        reader = PdfReader(buffer)
        pages = []
        for idx, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(f"[Page {idx + 1}]\n{txt}")
        return "\n\n".join(pages)
    if ext == "docx":
        doc = Document(buffer)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for t_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                rows.append(" | ".join(cell.text.strip() for cell in row.cells))
            paragraphs.append(f"[Table {t_idx + 1}]\n" + "\n".join(rows))
        return "\n\n".join(paragraphs)
    return buffer.read().decode("utf-8", errors="ignore")


def _dataframe_to_rag_documents(df: pd.DataFrame, filename: str) -> list[dict]:
    docs = []

    schema_lines = [f"File: {filename} | Total Rows: {len(df)} | Columns: {list(df.columns)}"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isna().sum())
        sample_vals = df[col].dropna().unique()[:5].tolist()
        schema_lines.append(
            f"- Column '{col}' ({dtype}, {nulls} nulls): sample values = {sample_vals}"
        )

    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        schema_lines.append("\nNumeric Summary Statistics:\n" + numeric_cols.describe().to_string())

    docs.append({
        "source": f"{filename} (Schema & Summary)",
        "content": "\n".join(schema_lines),
    })

    batch_size = 25
    max_rows = min(len(df), 12500)
    for start in range(0, max_rows, batch_size):
        end = min(start + batch_size, max_rows)
        subset = df.iloc[start:end]
        records_text = subset.to_json(orient="records", date_format="iso")
        docs.append({
            "source": f"{filename} (Rows {start + 1}–{end})",
            "content": f"Source: {filename}, Rows {start + 1} to {end}:\n{records_text}",
        })

    return docs


def _chunk_text(text: str, filename: str, chunk_size: int = 1000, overlap: int = 180) -> list[dict]:
    chunks = []
    start = 0
    idx = 1
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "source": f"{filename} (Chunk {idx})",
                "content": chunk,
            })
            idx += 1
        start += chunk_size - overlap
    return chunks