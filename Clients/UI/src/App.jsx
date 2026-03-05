import { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export default function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [showChunks, setShowChunks] = useState(false)

  const canAsk = useMemo(() => question.trim().length > 0 && !loading, [question, loading])

  async function askQuestion(e) {
    e.preventDefault()
    if (!canAsk) return

    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() })
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || `Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to get answer')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="container">
        <h1>RAG Question UI</h1>
        <p className="subtitle">Ask from browser instead of terminal.</p>

        <form onSubmit={askQuestion} className="ask-form">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask anything from your docs.txt context..."
            rows={4}
          />

          <button type="submit" disabled={!canAsk}>
            {loading ? 'Asking...' : 'Ask'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <section className="result">
            <h2>Answer</h2>
            <p>{result.answer}</p>

            {result.citations?.length > 0 && (
              <p className="citations">
                Sources: {result.citations.map((id) => `[${id}]`).join(', ')}
              </p>
            )}

            <button
              type="button"
              className="toggle"
              onClick={() => setShowChunks((v) => !v)}
            >
              {showChunks ? 'Hide Retrieved Chunks' : 'Show Retrieved Chunks'}
            </button>

            {showChunks && (
              <ul className="chunks">
                {result.retrieved_chunks?.map((chunk) => (
                  <li key={chunk.chunk_id}>
                    <div className="meta">
                      chunk={chunk.chunk_id} source={chunk.source_doc_id} score={Number(chunk.score).toFixed(3)}
                    </div>
                    <div>{chunk.text}</div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}
      </div>
    </div>
  )
}
