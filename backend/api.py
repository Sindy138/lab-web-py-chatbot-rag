"""
API FastAPI para el chatbot RAG de TechNova Solutions.
Arrancar: uvicorn api:app --reload --port 8000
Swagger UI: http://localhost:8000/docs
"""

import re
import time
import logging
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import chatbot

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting: max 10 peticiones / minuto por IP
# ---------------------------------------------------------------------------
RATE_LIMIT = 10
RATE_WINDOW = 60  # segundos

# {ip: [(timestamp, ...), ...]}
request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    window_start = now - RATE_WINDOW
    request_log[ip] = [t for t in request_log[ip] if t > window_start]
    if len(request_log[ip]) >= RATE_LIMIT:
        return False
    request_log[ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Detección de información personal
# ---------------------------------------------------------------------------
PERSONAL_DATA_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "email"),
    (re.compile(r"\b\d{8}[A-HJ-NP-TV-Z]\b"), "DNI/NIE"),
    (re.compile(r"\b\d{9}\b"), "teléfono"),
    (re.compile(r"\bmi nombre es\b|\bme llamo\b|\bsoy [A-ZÁÉÍÓÚ][a-záéíóú]+\b", re.IGNORECASE), "nombre personal"),
]


def detect_personal_info(text: str) -> list[str]:
    detected = []
    for pattern, label in PERSONAL_DATA_PATTERNS:
        if pattern.search(text):
            detected.append(label)
    return detected


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TechNova Chatbot API",
    description="Chatbot RAG sobre documentos de TechNova Solutions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    pregunta: str = Field(..., min_length=1, max_length=500, description="Pregunta del usuario (máx. 500 caracteres)")
    session_id: str = Field(default="default", description="ID de sesión para mantener el historial")


class ChatResponse(BaseModel):
    respuesta: str
    fuentes: list[str]
    session_id: str
    fragmentos_usados: int
    advertencia_privacidad: Optional[str] = None


class HistoryMessage(BaseModel):
    role: str
    content: str


# ---------------------------------------------------------------------------
# Middleware de logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "method=%s path=%s status=%d duration=%.3fs ip=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration,
        request.client.host if request.client else "unknown",
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, summary="Enviar una pregunta al chatbot")
async def post_chat(body: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"

    if not check_rate_limit(ip):
        logger.warning("Rate limit excedido para IP=%s", ip)
        raise HTTPException(
            status_code=429,
            detail="Demasiadas peticiones. Máximo 10 por minuto. Inténtalo de nuevo en un momento.",
        )

    advertencia = None
    personal_detected = detect_personal_info(body.pregunta)
    if personal_detected:
        tipos = ", ".join(personal_detected)
        advertencia = (
            f"Tu pregunta parece contener información personal ({tipos}). "
            "Se recomienda no incluir datos personales en consultas al chatbot."
        )
        logger.warning("Información personal detectada en pregunta. session_id=%s tipos=%s", body.session_id, tipos)

    logger.info(
        "Chat request: session_id=%s pregunta_len=%d fragmentos_esperados=3",
        body.session_id,
        len(body.pregunta),
    )

    try:
        result = chatbot.chat(body.pregunta, body.session_id)
    except Exception as exc:
        logger.error("Error en chatbot.chat: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Error interno del chatbot: {str(exc)}")

    return ChatResponse(
        respuesta=result["respuesta"],
        fuentes=result["fuentes"],
        session_id=result["session_id"],
        fragmentos_usados=result["fragmentos_usados"],
        advertencia_privacidad=advertencia,
    )


@app.get("/chat/history/{session_id}", response_model=list[HistoryMessage], summary="Obtener historial de sesión")
async def get_history(session_id: str):
    history = chatbot.get_history(session_id)
    return [HistoryMessage(role=m["role"], content=m["content"]) for m in history]


@app.get("/documentos", response_model=list[str], summary="Listar documentos indexados")
async def get_documentos():
    docs = chatbot.list_documents()
    if not docs:
        raise HTTPException(status_code=404, detail="No hay documentos indexados. Ejecuta indexer.py primero.")
    return docs


@app.get("/", summary="Health check")
async def root():
    return {"status": "ok", "mensaje": "TechNova Chatbot API funcionando. Visita /docs para la documentación."}


@app.get("/health", summary="Estado del servicio")
async def health():
    docs = chatbot.list_documents()
    return {
        "status": "ok",
        "documentos_indexados": len(docs),
        "modelo_chat": chatbot.CHAT_MODEL,
        "modelo_embeddings": chatbot.EMBEDDING_MODEL,
        "lm_studio_url": chatbot.BASE_URL,
    }
