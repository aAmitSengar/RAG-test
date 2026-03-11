import { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

function UserMessage({ text }) {
  return (
    <div className="msg-row msg-row--user">
      <div className="bubble bubble--user">{text}</div>
      <div className="avatar avatar--user">You</div>
    </div>
  )
}

function AssistantMessage({ data, error }) {
  const [showChunks, setShowChunks] = useState(false)

  if (error) {
    return (
      <div className="msg-row msg-row--assistant">
        <div className="avatar avatar--assistant">AI</div>
        <div className="bubble bubble--error">{error}</div>
      </div>
    )
  }

  return (
    <div className="msg-row msg-row--assistant">
      <div className="avatar avatar--assistant">AI</div>
      <div className="bubble bubble--assistant">
        <p className="bubble-text">{data.answer}</p>

        {data.citations?.length > 0 && (
          <p className="citations">
            Sources: {data.citations.map((id) => `[${id}]`).join(', ')}
          </p>
        )}

        {data.retrieved_chunks?.length > 0 && (
          <>
            <button
              type="button"
              className="toggle"
              onClick={() => setShowChunks((v) => !v)}
            >
              {showChunks ? 'Hide Retrieved Chunks' : 'Show Retrieved Chunks'}
            </button>

            {showChunks && (
              <ul className="chunks">
                {data.retrieved_chunks.map((chunk) => (
                  <li key={chunk.chunk_id}>
                    <div className="meta">
                      chunk={chunk.chunk_id} source={chunk.source_doc_id} score={Number(chunk.score).toFixed(3)}
                    </div>
                    <div>{chunk.text}</div>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="msg-row msg-row--assistant">
      <div className="avatar avatar--assistant">AI</div>
      <div className="bubble bubble--assistant bubble--typing">
        <span className="dot" /><span className="dot" /><span className="dot" />
      </div>
    </div>
  )
}

export default function App() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState([])
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canSend) sendMessage()
    }
  }

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return

    setMessages((prev) => [...prev, { id: Date.now(), role: 'user', text }])
    setInput('')
    setLoading(true)
    textareaRef.current?.focus()

    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text })
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || `Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setMessages((prev) => [...prev, { id: Date.now() + 1, role: 'assistant', data }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, role: 'assistant', error: err.message || 'Failed to get answer' }
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-layout">
      <header className="chat-header">
        <div className="chat-header__icon">🤖</div>
        <div>
          <div className="chat-header__title">RAG Assistant</div>
          <div className="chat-header__sub">Powered by your documents</div>
        </div>
      </header>

      <main className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-state__icon">💬</div>
            <p>Ask me anything from your docs.</p>
          </div>
        )}

        {messages.map((msg) =>
          msg.role === 'user' ? (
            <UserMessage key={msg.id} text={msg.text} />
          ) : (
            <AssistantMessage key={msg.id} data={msg.data} error={msg.error} />
          )
        )}

        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </main>

      <footer className="chat-input-bar">
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything from your docs… (Enter to send, Shift+Enter for new line)"
          rows={1}
        />
        <button
          type="button"
          className="send-btn"
          disabled={!canSend}
          onClick={sendMessage}
          aria-label="Send"
        >
          ➤
        </button>
      </footer>
    </div>
  )
}
