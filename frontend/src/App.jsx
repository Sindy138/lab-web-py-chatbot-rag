import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_BASE = "";
const DEFAULT_SESSION = `session-${Date.now()}`;

function generateSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hola, soy el asistente de TechNova Solutions. Puedo responder preguntas sobre nuestras políticas de RRHH, producto, precios, soporte técnico y cultura de empresa. ¿En qué puedo ayudarte?",
      fuentes: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(DEFAULT_SESSION);
  const [documents, setDocuments] = useState([]);
  const [docsError, setDocsError] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [apiStatus, setApiStatus] = useState("checking");
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    checkHealth();
    fetchDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) setApiStatus("ok");
      else setApiStatus("error");
    } catch {
      setApiStatus("error");
    }
  }

  async function fetchDocuments() {
    try {
      const res = await fetch(`${API_BASE}/documentos`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      } else {
        setDocsError(true);
      }
    } catch {
      setDocsError(true);
    }
  }

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;
    if (question.length > 500) {
      alert("La pregunta no puede superar los 500 caracteres.");
      return;
    }

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta: question, session_id: sessionId }),
      });

      if (res.status === 429) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Has superado el límite de 10 peticiones por minuto. Por favor, espera un momento.",
            fuentes: [],
            error: true,
          },
        ]);
        return;
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Error en la API");
      }

      const data = await res.json();
      const botMessage = {
        role: "assistant",
        content: data.respuesta,
        fuentes: data.fuentes || [],
        fragmentos: data.fragmentos_usados,
        advertencia: data.advertencia_privacidad,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error al conectar con la API: ${err.message}. Asegúrate de que el backend está corriendo en el puerto 8000.`,
          fuentes: [],
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function resetSession() {
    const newId = generateSessionId();
    setSessionId(newId);
    setMessages([
      {
        role: "assistant",
        content: "Nueva conversación iniciada. ¿En qué puedo ayudarte?",
        fuentes: [],
      },
    ]);
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen((v) => !v)} title="Documentos">
            <MenuIcon />
          </button>
          <div className="logo">
            <span className="logo-icon">TN</span>
            <span className="logo-text">TechNova Chatbot</span>
          </div>
        </div>
        <div className="header-right">
          <div className={`status-dot ${apiStatus}`} title={`API: ${apiStatus}`} />
          <span className="status-label">{apiStatus === "ok" ? "API conectada" : "API desconectada"}</span>
          <button className="btn-secondary" onClick={resetSession}>Nueva sesión</button>
        </div>
      </header>

      <div className="main">
        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="sidebar">
            <div className="sidebar-header">
              <h2>Documentos</h2>
              <span className="badge">{documents.length}</span>
            </div>
            {docsError ? (
              <p className="sidebar-error">No se pudo cargar la lista.<br />Ejecuta <code>indexer.py</code> primero.</p>
            ) : documents.length === 0 ? (
              <p className="sidebar-empty">Cargando documentos...</p>
            ) : (
              <ul className="doc-list">
                {documents.map((doc) => (
                  <li key={doc} className="doc-item">
                    <FileIcon />
                    <span>{doc}</span>
                  </li>
                ))}
              </ul>
            )}
            <div className="sidebar-footer">
              <p className="session-label">Sesión activa:</p>
              <code className="session-id">{sessionId.slice(-12)}</code>
            </div>
          </aside>
        )}

        {/* Chat */}
        <div className="chat-container">
          <div className="messages">
            {messages.map((msg, i) => (
              <ChatMessage key={i} message={msg} />
            ))}
            {loading && (
              <div className="message bot">
                <div className="avatar bot-avatar">TN</div>
                <div className="bubble bot-bubble loading-bubble">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="input-area">
            <div className="input-wrapper">
              <textarea
                ref={inputRef}
                className="input-field"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Escribe tu pregunta sobre TechNova... (Enter para enviar)"
                rows={2}
                maxLength={500}
                disabled={loading}
              />
              <div className="input-footer">
                <span className={`char-count ${input.length > 450 ? "warning" : ""}`}>
                  {input.length}/500
                </span>
                <button
                  className="btn-send"
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                >
                  <SendIcon />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatMessage({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "bot"}`}>
      {!isUser && <div className="avatar bot-avatar">TN</div>}
      <div className="message-body">
        {message.advertencia && (
          <div className="privacy-warning">
            <WarningIcon /> {message.advertencia}
          </div>
        )}
        <div className={`bubble ${isUser ? "user-bubble" : "bot-bubble"} ${message.error ? "error-bubble" : ""}`}>
          <p className="message-text">{message.content}</p>
        </div>
        {!isUser && message.fuentes && message.fuentes.length > 0 && (
          <div className="sources">
            <span className="sources-label">Fuentes:</span>
            {message.fuentes.map((f) => (
              <span key={f} className="source-chip">{f}</span>
            ))}
            {message.fragmentos !== undefined && (
              <span className="fragments-count">{message.fragmentos} fragmentos</span>
            )}
          </div>
        )}
      </div>
      {isUser && <div className="avatar user-avatar">Tú</div>}
    </div>
  );
}

function MenuIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ display: "inline", marginRight: 4 }}>
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
