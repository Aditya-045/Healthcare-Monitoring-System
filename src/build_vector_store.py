"""
Week 3, Days 11-12 — build the RAG vector store.

Run locally (NOT executed here):
    python src/build_vector_store.py

What it does:
  1. Reads every data/reference/*.md file.
  2. Splits each file into chunks along its "## " headings (one chunk per
     lab-marker section — a natural, self-contained unit of meaning).
  3. Embeds each chunk with a small local sentence-transformer model
     (all-MiniLM-L6-v2 — downloaded once, then cached locally by the library).
  4. Stores the chunks + embeddings + metadata (disease, section title) in a
     persistent local ChromaDB collection under vector_store/.

Re-run this any time the reference .md files change; it recreates the
collection from scratch (safe to run repeatedly).
"""
import argparse
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "medical_reference"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Which reference file maps to which disease key (must match app.py's disease keys)
FILE_TO_DISEASE = {
    "kidney_reference.md": "kidney",
    "liver_reference.md": "liver",
}


def chunk_markdown(text):
    """
    Splits a markdown file into (title, body) chunks on '## ' headings.
    The '# ' top-level title is dropped (it's just the file title, not a
    retrievable fact on its own); the 'Overview' and similar '##' sections
    are kept like any other chunk.
    """
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        lines = section.splitlines()
        title = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            chunks.append((title, body))
    return chunks


def main(args):
    reference_dir = Path(args.reference_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(reference_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {reference_dir}")

    print(f"Found {len(md_files)} reference file(s): {[f.name for f in md_files]}")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Fresh rebuild each run so stale chunks never linger after edits
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    ids, documents, metadatas = [], [], []
    for md_file in md_files:
        disease = FILE_TO_DISEASE.get(md_file.name, "unknown")
        text = md_file.read_text()
        chunks = chunk_markdown(text)
        print(f"  {md_file.name}: {len(chunks)} chunks (disease={disease})")
        for i, (title, body) in enumerate(chunks):
            ids.append(f"{md_file.stem}_{i}")
            documents.append(f"{title}\n{body}")
            metadatas.append({"disease": disease, "title": title, "source": md_file.name})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"\nStored {len(ids)} chunks in collection '{COLLECTION_NAME}' at {persist_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RAG vector store from reference markdown")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--persist-dir", default="vector_store")
    main(parser.parse_args())
