"""
ingest.py — Load documents from docs/ and index them into ChromaDB.
Run this once (or whenever you update your docs):
    python ingest.py
"""

import os
import glob
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv()
DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"
EMBED_MODEL = "text-embedding-3-small"  # cheap & good
CHUNK_SIZE = 300        # characters per chunk
CHUNK_OVERLAP = 50      # overlap to avoid splitting mid-sentence

# ── Helpers ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")

    openai_client = OpenAI(api_key=api_key)

    # ── ChromaDB ──
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh start — delete existing collection if present
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Load and chunk documents ──
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not doc_files:
        print(f"⚠️  No .txt files found in '{DOCS_DIR}'. Add some documents first.")
        return

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadata: list[dict] = []

    for filepath in doc_files:
        filename = os.path.basename(filepath)
        print(f"📄 Processing: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}::chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({"source": filename, "chunk_index": i})

    # ── Embed and store ──
    print(f"\n⚡ Embedding {len(all_chunks)} chunks (model: {EMBED_MODEL})...")
    BATCH = 50  # OpenAI max batch is 2048 inputs but keep it sane
    for start in range(0, len(all_chunks), BATCH):
        batch_chunks = all_chunks[start : start + BATCH]
        batch_ids = all_ids[start : start + BATCH]
        batch_meta = all_metadata[start : start + BATCH]
        embeddings = get_embeddings(openai_client, batch_chunks)
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_meta,
        )
        print(f"   ✓ {min(start + BATCH, len(all_chunks))} / {len(all_chunks)} chunks stored")

    print(f"\n✅  Ingested {len(all_chunks)} chunks from {len(doc_files)} file(s) into ChromaDB.")
    print(f"   Collection '{COLLECTION_NAME}' is saved at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
