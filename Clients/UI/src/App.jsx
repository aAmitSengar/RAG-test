import { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export default function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [showChunks, setShowChunks] = useState(false)

  // Feedback state
  const [feedbackState, setFeedbackState] = useState('idle') // idle | wrong | submitting | done
  const [correction, setCorrection] = useState('')
  const [feedbackMessage, setFeedbackMessage] = useState('')

  const canAsk = useMemo(() => question.trim().length > 0 && !loading, [question, loading])

  async function askQuestion(e) {
    e.preventDefault()
    if (!canAsk) return

    setLoading(true)
    setError('')
    // Reset feedback whenever a new question is asked.
    setFeedbackState('idle')
    setCorrection('')
    setFeedbackMessage('')

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

  async function submitCorrection(e) {
    e.preventDefault()
    if (!correction.trim() || !result) return

    setFeedbackState('submitting')
    try {
      const response = await fetch(`${API_BASE}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: result.query,
          wrong_answer: result.answer,
          correct_answer: correction.trim()
        })
      })

      const payload = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(payload.detail || `Feedback failed with status ${response.status}`)
      }

      setFeedbackMessage(payload.message || 'Correction saved!')
      setFeedbackState('done')
    } catch (err) {
      setFeedbackMessage(err.message || 'Failed to save correction.')
      setFeedbackState('idle')
    }
  }

  return (
    <div className="page">
      <div className="container">
        <h1>RAG Question UI</h1>
        <p className="subtitle">Ask questions — and teach the system when it gets something wrong.</p>

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

            {/* ── Feedback panel ── */}
            {feedbackState !== 'done' && (
              <div className="feedback-row">
                <span className="feedback-label">Was this answer correct?</span>

                <button
                  type="button"
                  className="feedback-btn correct"
                  onClick={() => { setFeedbackState('idle'); setFeedbackMessage(''); }}
                  title="Yes, this is correct"
                >
                  👍 Yes
                </button>

                <button
                  type="button"
                  className={`feedback-btn wrong${feedbackState === 'wrong' ? ' active' : ''}`}
                  onClick={() => setFeedbackState('wrong')}
                  title="No, this is wrong — I'll provide the correct answer"
                >
                  👎 No — teach it
                </button>
              </div>
            )}

            {feedbackState === 'wrong' && (
              <form onSubmit={submitCorrection} className="correction-form">
                <label htmlFor="correction">What is the correct answer?</label>
                <textarea
                  id="correction"
                  value={correction}
                  onChange={(e) => setCorrection(e.target.value)}
                  placeholder="Type the correct answer here..."
                  rows={3}
                />
                <div className="correction-actions">
                  <button
                    type="submit"
                    disabled={!correction.trim()}
                  >
                    Submit correction
                  </button>
                  <button
                    type="button"
                    className="cancel-btn"
                    onClick={() => { setFeedbackState('idle'); setCorrection(''); }}
                  >
                    Cancel
                  </button>
                </div>
              </form>
            )}

            {feedbackState === 'submitting' && (
              <p className="feedback-status">Saving correction and updating knowledge base…</p>
            )}

            {feedbackState === 'done' && (
              <p className="feedback-status success">{feedbackMessage}</p>
            )}

            {feedbackState === 'idle' && feedbackMessage && (
              <p className="feedback-status error-msg">{feedbackMessage}</p>
            )}

            {/* ── Retrieved chunks toggle ── */}
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
