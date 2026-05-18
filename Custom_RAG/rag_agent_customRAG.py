"""
rag_agent.py — CLI RAG agent using ChromaDB + OpenAI.
Run after ingestion:
    python rag_agent.py
Type your question and press Enter. Type 'exit' or 'quit' to stop.
"""

import os
import sys
import textwrap
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv()
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"
EMBED_MODEL = "text-embedding-3-small"   # cheap embeddings
CHAT_MODEL = "gpt-4o-mini"              # cheap generation
TOP_K = 3                               # number of chunks to retrieve
MAX_CONTEXT_CHARS = 3000                # safety limit for context length

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using ONLY the provided context.
If the answer is not found in the context, say: "I don't have information about that in my knowledge base."
Always be concise and direct. When possible, mention which source document your answer came from.
"""

# SYSTEM_PROMPT = """\
# You are a helpful assistant. Use the provided context to answer questions, but feel free to greet the user normally.
# """

# ── Helpers ──────────────────────────────────────────────────────────────────

def embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


def retrieve(collection, query_embedding: list[float], top_k: int = TOP_K):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metas, distances))


def build_context(hits: list) -> str:
    parts = []
    total = 0
    for doc, meta, dist in hits:
        snippet = doc[:MAX_CONTEXT_CHARS - total]
        parts.append(f"[Source: {meta['source']}]\n{snippet}")
        total += len(snippet)
        if total >= MAX_CONTEXT_CHARS:
            break
    return "\n\n---\n\n".join(parts)


def generate_answer(client: OpenAI, question: str, context: str) -> str:
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def print_separator(char="─", width=60):
    print(char * width)


def wrap_print(text: str, width: int = 70, indent: str = "  "):
    for line in text.splitlines():
        if line.strip():
            wrapped = textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            print(wrapped)
        else:
            print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌  OPENAI_API_KEY not found in .env — cannot start.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)

    # ── Load ChromaDB ──
    if not os.path.exists(CHROMA_DIR):
        print("❌  ChromaDB not found. Run 'python ingest.py' first.")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"❌  Collection '{COLLECTION_NAME}' not found. Run 'python ingest.py' first.")
        sys.exit(1)

    doc_count = collection.count()

    # ── Banner ──
    print()
    print_separator("═")
    print("  🤖  RAG Demo Agent")
    print(f"  Model : {CHAT_MODEL}  |  Embeddings : {EMBED_MODEL}")
    print(f"  Vector store : ChromaDB  |  Chunks indexed : {doc_count}")
    print_separator("═")
    print("  Type your question and press Enter.")
    print("  Type  'exit'  or  'quit'  to stop.")
    print_separator("─")
    print()

    # ── Chat loop ──
    while True:
        try:
            query = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋  Goodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("\n👋  Goodbye!")
            break

        print()
        print("🔍  Retrieving relevant chunks...")

        try:
            query_embedding = embed_query(openai_client, query)
            hits = retrieve(collection, query_embedding)
        except Exception as e:
            print(f"❌  Retrieval error: {e}\n")
            continue

        # Show sources
        print_separator()
        print("📚  Top sources retrieved:")
        for i, (doc, meta, dist) in enumerate(hits, 1):
            similarity = 1 - dist  # cosine distance → similarity
            preview = doc[:80].replace("\n", " ") + ("..." if len(doc) > 80 else "")
            print(f"  [{i}] {meta['source']}  (similarity: {similarity:.2f})")
            print(f"      \"{preview}\"")
        print_separator()

        # Generate answer
        print("💬  Generating answer...\n")
        try:
            context = build_context(hits)
            answer = generate_answer(openai_client, query, context)
        except Exception as e:
            print(f"❌  Generation error: {e}\n")
            continue

        print("Agent>")
        wrap_print(answer)
        print()
        print_separator("─")
        print()


if __name__ == "__main__":
    main()
