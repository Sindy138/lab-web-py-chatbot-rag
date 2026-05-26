"""
Chatbot RAG: recupera fragmentos relevantes de ChromaDB y genera respuestas con LM Studio.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb import EmbeddingFunction, Embeddings

load_dotenv()

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", "local-model")
EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "technova_docs"
TOP_K = 3
MAX_HISTORY = 10

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Historial de sesiones en memoria: {session_id: [{"role": ..., "content": ...}]}
session_histories: dict[str, list[dict]] = {}


class LMStudioEmbeddings(EmbeddingFunction):
    def __call__(self, input: list[str]) -> Embeddings:
        embeddings = []
        for text in input:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings


def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=LMStudioEmbeddings(),
    )


def retrieve_fragments(question: str, top_k: int = TOP_K) -> tuple[list[str], list[str]]:
    """Devuelve (textos, fuentes) de los fragmentos más relevantes."""
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    sources = list(dict.fromkeys(m["source"] for m in metadatas))
    return documents, sources


SYSTEM_PROMPT = """Eres el asistente virtual de TechNova Solutions.
Responde ÚNICAMENTE usando la información del contexto proporcionado.
Si el contexto no contiene información suficiente para responder la pregunta, di exactamente:
"No tengo información sobre eso en los documentos disponibles."
No inventes datos, cifras, políticas ni información que no aparezca en el contexto.
Sé conciso, claro y profesional. Responde siempre en español."""


def chat(pregunta: str, session_id: str) -> dict:
    """
    1. Recupera los TOP_K fragmentos más relevantes.
    2. Construye el prompt con contexto.
    3. Mantiene historial de conversación por session_id.
    4. Retorna respuesta con fuentes y metadatos.
    """
    fragments, sources = retrieve_fragments(pregunta)

    if fragments:
        context = "\n\n---\n\n".join(fragments)
        context_block = f"CONTEXTO DISPONIBLE:\n{context}"
    else:
        context_block = "CONTEXTO DISPONIBLE: (ninguno)"

    if session_id not in session_histories:
        session_histories[session_id] = []

    history = session_histories[session_id]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_block},
        {"role": "assistant", "content": "Entendido. Usaré únicamente ese contexto para responder."},
    ]
    messages += history[-MAX_HISTORY:]
    messages.append({"role": "user", "content": pregunta})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    history.append({"role": "user", "content": pregunta})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY * 2:
        session_histories[session_id] = history[-(MAX_HISTORY * 2):]

    return {
        "respuesta": answer,
        "fuentes": sources,
        "session_id": session_id,
        "fragmentos_usados": len(fragments),
    }


def get_history(session_id: str) -> list[dict]:
    return session_histories.get(session_id, [])


def list_documents() -> list[str]:
    """Devuelve la lista de documentos indexados."""
    try:
        collection = get_collection()
        result = collection.get(include=["metadatas"])
        sources = list(dict.fromkeys(m["source"] for m in result["metadatas"]))
        return sources
    except Exception:
        return []


if __name__ == "__main__":
    print("Chatbot TechNova (escribe 'salir' para terminar)\n")
    sid = "consola-001"
    while True:
        q = input("Tú: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        if not q:
            continue
        result = chat(q, sid)
        print(f"\nAsistente: {result['respuesta']}")
        if result["fuentes"]:
            print(f"Fuentes: {', '.join(result['fuentes'])}")
        print()
