import numpy as np
import faiss
from fastembed import TextEmbedding
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


class HybridRAGStore:
    def __init__(self):
        self.embedder = get_embedding_model()
        self.index = None
        self.documents = []

    def build_index(self, documents: list[dict]):
        self.documents = documents
        if not documents:
            self.index = None
            return

        texts = [doc["content"] for doc in documents]
        embeddings = list(self.embedder.embed(texts))
        matrix = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.documents:
            return []

        q_emb = list(self.embedder.embed([query]))
        q_vec = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        return results