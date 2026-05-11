import React, { useMemo, useState } from 'react'
import api from '../api'
import ResultCard from '../components/ResultCard'

export default function Text() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')

  const previewInput = useMemo(() => {
    const clean = text.trim().replace(/\s+/g, ' ')
    if (!clean) return 'No text entered'
    if (clean.length <= 90) return clean
    return `${clean.slice(0, 87)}...`
  }, [text])

  const charCount = useMemo(() => {
    return text.length
  }, [text])

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)
    setStatus('')
    if (!text.trim()) {
      setError('Please enter transcript text before submitting.')
      return
    }
    if (text.length > 5000) {
      setError('Please keep text under 5000 characters.')
      return
    }

    setLoading(true)
    setStatus('Running text emotion inference on backend...')
    try {
      const res = await api.postText(text)
      setResult(res)
      setStatus('Prediction complete.')
    } catch (err) {
      setError(err.message || 'Unable to predict text emotion right now.')
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="speech-page">
      {/* Hero Section */}
      <section style={{ marginBottom: '40px' }}>
        <div className="info-badge">📝 Text Analysis</div>
        <h1>Text Emotion Recognition</h1>
        <p style={{ fontSize: '1.05rem', maxWidth: '600px' }}>
          Enter text to analyze emotional sentiment using NLP and machine learning. Get instant emotion classification with confidence scores and probability distribution.
        </p>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px', marginBottom: '40px' }}>
        {/* Input Section */}
        <section className="card">
          <h2 style={{ marginBottom: '20px' }}>✍️ Enter Text</h2>
          <p style={{ marginBottom: '24px' }}>
            Type or paste text to analyze emotional sentiment and get predictions.
          </p>

          <form onSubmit={onSubmit}>
            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <label className="form-label">Your Text</label>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  {charCount} / 5000
                </span>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={7}
                maxLength={5000}
                placeholder="Type something like: I feel amazing today, or I'm really sad about this..."
                style={{
                  width: '100%',
                  padding: '14px 16px',
                  borderRadius: '10px',
                  border: '1px solid rgba(102, 126, 234, 0.2)',
                  background: 'rgba(15, 23, 42, 0.6)',
                  color: 'var(--text)',
                  fontFamily: "'Inter', sans-serif",
                  fontSize: '1rem',
                  resize: 'vertical',
                  minHeight: '180px',
                  transition: 'all 0.3s ease'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = 'rgba(102, 126, 234, 0.5)'
                  e.target.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)'
                  e.target.style.background = 'rgba(15, 23, 42, 0.8)'
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(102, 126, 234, 0.2)'
                  e.target.style.boxShadow = 'none'
                  e.target.style.background = 'rgba(15, 23, 42, 0.6)'
                }}
              />
            </div>

            {text.trim() && (
              <div style={{
                background: 'rgba(79, 172, 254, 0.1)',
                border: '1px solid rgba(79, 172, 254, 0.3)',
                borderRadius: '10px',
                padding: '12px 16px',
                marginBottom: '20px',
                fontSize: '0.9rem',
              }}>
                <p style={{ margin: 0, fontWeight: '500', color: 'var(--text-secondary)' }}>Preview</p>
                <p style={{ margin: '8px 0 0', color: 'var(--text-muted)' }}>{previewInput}</p>
              </div>
            )}

            {status && (
              <div style={{
                background: 'rgba(79, 172, 254, 0.15)',
                border: '1px solid var(--primary)',
                borderRadius: '10px',
                padding: '12px 16px',
                marginBottom: '20px',
                color: 'var(--accent-light)',
                fontSize: '0.9rem',
                fontWeight: '500'
              }}>
                ⏳ {status}
              </div>
            )}

            {error && (
              <div style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid var(--error)',
                borderRadius: '10px',
                padding: '12px 16px',
                marginBottom: '20px',
                color: '#fecdd3'
              }}>
                ⚠️ {error}
              </div>
            )}

            <button 
              className="btn btn-primary" 
              type="submit" 
              disabled={loading || !text.trim()}
              style={{ width: '100%' }}
            >
              {loading ? '⏳ Analyzing...' : '🚀 Analyze Emotion'}
            </button>
          </form>
        </section>

        {/* Results Section */}
        <section className="card">
          <h2 style={{ marginBottom: '20px' }}>📊 Prediction Result</h2>
          
          {loading && (
            <div className="empty-state">
              <div className="loading-spinner"></div>
              <p>Running NLP sentiment analysis on your text...</p>
            </div>
          )}

          {!loading && result && (
            <>
              <ResultCard title="Text Prediction" fileName={previewInput} payload={result} />
              <div style={{
                marginTop: '20px',
                paddingTop: '16px',
                borderTop: '1px solid rgba(102, 126, 234, 0.2)',
                display: 'grid',
                gap: '12px'
              }}>
                <div>
                  <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '4px' }}>Cleaned Text</p>
                  <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{result.cleaned_text || 'N/A'}</p>
                </div>
                <div>
                  <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '4px' }}>Token Count</p>
                  <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{result.token_count ?? 'N/A'} tokens</p>
                </div>
              </div>
            </>
          )}

          {!loading && !result && (
            <div className="empty-state">
              <p style={{ fontSize: '1.05rem', color: 'var(--text-muted)' }}>
                👆 Enter text to see emotion prediction results
              </p>
            </div>
          )}
        </section>
      </div>

      {/* Info Cards */}
      <section style={{ marginTop: '40px' }}>
        <h2 style={{ marginBottom: '24px', textAlign: 'center' }}>How It Works</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>1️⃣</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Enter Text</h3>
            <p style={{ fontSize: '0.9rem' }}>Type or paste text content you want to analyze for emotional tone</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>2️⃣</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Process</h3>
            <p style={{ fontSize: '0.9rem' }}>ML model extracts features using TF-IDF and runs classification inference</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>3️⃣</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Results</h3>
            <p style={{ fontSize: '0.9rem' }}>Get emotion prediction, confidence score, and probability distribution</p>
          </div>
        </div>
      </section>
    </div>
  )
}

