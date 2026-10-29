"""
Week 3, Day 13 — retrieval function on top of the vector store built by
build_vector_store.py.

Standalone test (NOT executed here):
    python src/rag.py --query "what does high creatinine mean" --disease kidney
"""
import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "medical_reference"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_collection(persist_dir="vector_store"):
    """
    Loads the persistent Chroma collection. Must use the SAME embedding
    model as build_vector_store.py or query embeddings won't be comparable
    to the stored ones.
    Returns None if the vector store hasn't been built yet.
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return None
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(persist_path))
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        return None


def retrieve(query, disease=None, top_k=4, persist_dir="vector_store"):
    """
    Returns a list of {"title": ..., "text": ..., "disease": ...} dicts,
    most relevant first. Filters to a single disease's chunks when `disease`
    is given (recommended — keeps kidney facts out of a liver conversation
    and vice versa).
    """
    collection = get_collection(persist_dir)
    if collection is None:
        return []

    where = {"disease": disease} if disease else None
    results = collection.query(query_texts=[query], n_results=top_k, where=where)

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(documents, metadatas):
        chunks.append({
            "title": meta.get("title", ""),
            "text": doc,
            "disease": meta.get("disease", ""),
        })
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test retrieval against the RAG vector store")
    parser.add_argument("--query", required=True)
    parser.add_argument("--disease", default=None, choices=[None, "kidney", "liver"])
    parser.add_argument("--top-k", dest="top_k", type=int, default=4)
    parser.add_argument("--persist-dir", default="vector_store")
    args = parser.parse_args()

    results = retrieve(args.query, disease=args.disease, top_k=args.top_k, persist_dir=args.persist_dir)
    if not results:
        print("No results (vector store missing/empty, or run build_vector_store.py first).")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['title']}  (disease={r['disease']})")
        print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
