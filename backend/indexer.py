"""
Indexer: lee docs/*.txt, crea embeddings con LM Studio y los almacena en ChromaDB.
Uso: python indexer.py
"""

import os
import glob
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb import EmbeddingFunction, Embeddings

load_dotenv()

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "technova_docs"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


class LMStudioEmbeddings(EmbeddingFunction):
    def __call__(self, input: list[str]) -> Embeddings:
        embeddings = []
        for text in input:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def index_documents():
    txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not txt_files:
        print(f"No se encontraron archivos .txt en {DOCS_DIR}")
        return

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Colección '{COLLECTION_NAME}' eliminada para re-indexar.")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LMStudioEmbeddings(),
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    total_tokens = 0
    all_ids = []
    all_texts = []
    all_metadatas = []

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)
        doc_tokens = count_tokens(content)
        total_tokens += doc_tokens

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}__chunk_{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append({"source": filename, "chunk_id": i, "total_chunks": len(chunks)})
            total_chunks += 1

        print(f"  [{filename}] {len(chunks)} chunks | {doc_tokens} tokens")

    print(f"\nGenerando embeddings para {total_chunks} chunks...")
    batch_size = 10
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_texts = all_texts[i : i + batch_size]
        batch_metas = all_metadatas[i : i + batch_size]
        collection.add(documents=batch_texts, metadatas=batch_metas, ids=batch_ids)
        print(f"  Lote {i // batch_size + 1}/{(len(all_ids) - 1) // batch_size + 1} procesado.")

    cost_estimate = (total_tokens / 1000) * 0.0001
    print("\n========== RESUMEN ==========")
    print(f"Documentos indexados : {len(txt_files)}")
    print(f"Chunks generados     : {total_chunks}")
    print(f"Tokens procesados    : {total_tokens}")
    print(f"Coste estimado       : ${cost_estimate:.4f} USD")
    print(f"Colección ChromaDB   : {COLLECTION_NAME}")
    print(f"Directorio de datos  : {CHROMA_DIR}")
    print("=============================\n")


if __name__ == "__main__":
    print("Iniciando indexación de documentos...\n")
    index_documents()
    print("Indexación completada.")
