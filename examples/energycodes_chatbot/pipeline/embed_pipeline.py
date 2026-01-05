import argparse
import json
import os
import sys
import uuid

import chromadb
from langchain_ollama import OllamaEmbeddings


def embed_texts(texts, embedder):
    """Embed a list of strings using LangChain Ollama embeddings."""
    return embedder.embed_documents(texts)


def main():
    parser = argparse.ArgumentParser(description="Import Q/A JSON into ChromaDB (Ollama embeddings)")
    parser.add_argument("--json_path", default="./combined_outputs_v2.json", help="Path to JSON file")
    parser.add_argument("--collection", default="helpdesk_qna", help="Chroma collection name")
    parser.add_argument("--persist-dir", default="./chroma_persist", help="Chroma persistent directory")
    parser.add_argument("--embed-model", default="mxbai-embed-large",
                        help="Ollama embedding model name")
    args = parser.parse_args()

    # --- Load JSON ---
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: JSON must contain a list of items.")
        sys.exit(1)

    entries = []
    for item in data:
        if isinstance(item, dict) and "question" in item:
            question = str(item["question"])
            answer = str(item.get("answer", ""))
            entries.append({"question": question, "answer": answer})

    if not entries:
        print("No valid entries found.")
        sys.exit(0)

    # --- Initialize Ollama Embeddings ---
    embedder = OllamaEmbeddings(model=args.embed_model)

    # --- Initialize ChromaDB persistent client (1.2.2) ---
    client = chromadb.PersistentClient(path=args.persist_dir)

    # --- Get or create collection ---
    if args.collection in [c.name for c in client.list_collections()]:
        collection = client.get_collection(args.collection)
    else:
        collection = client.create_collection(args.collection)

    # --- Prepare embedding requests ---
    questions = [e["question"] for e in entries]
    answers = [e["answer"] for e in entries]

    # --- Embed all questions ---
    try:
        embeddings = embed_texts(questions, embedder)
    except Exception as e:
        print("Embedding error:", e)
        sys.exit(1)

    # --- Prepare insert data ---
    ids = [str(uuid.uuid4()) for _ in entries]
    metadatas = [{"answer": ans, "source_file": os.path.abspath(args.json_path)}
                 for ans in answers]

    # --- Insert into collection ---
    collection.add(
        ids=ids,
        documents=questions,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Successfully imported {len(entries)} entries into ChromaDB (collection='{args.collection}').")
    print(f"Persistent directory: {args.persist_dir}")


if __name__ == "__main__":
    main()



