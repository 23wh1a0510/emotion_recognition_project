import React, { useMemo, useState } from 'react'
import api from '../api'
import ResultCard from '../components/ResultCard'

export default function Multimodal() {
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')

  const fileLabel = useMemo(() => {
    if (!file) return 'No audio file selected'
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2)
    return `${file.name} · ${sizeMb} MB`
  }, [file])

  const isValidAudio = (selectedFile) => {
    if (!selectedFile) return false
    if (selectedFile.type && selectedFile.type.startsWith('audio/')) return true
    return /\.(wav|mp3|m4a|ogg|flac|aac)$/i.test(selectedFile.name)
  }

  const handleFileChange = (e) => {
    const sel = e.target.files?.[0] || null
    setError('')
    setResult(null)
    setFile(sel)
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)
    setStatus('')
    if (!file) return setError('Please select an audio file')
    if (!text.trim()) return setError('Please enter text')
    if (!isValidAudio(file)) return setError('Unsupported audio file type')

    setLoading(true)
    setStatus('Running speech and text models, then fusing results...')
    try {
      const res = await api.postMultimodal(file, text)
      setResult(res)
      setStatus('Multimodal prediction complete.')
    } catch (err) {
      setError(err.message || 'Multimodal inference failed')
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="speech-page">
      {/* Hero Section */}
      <section style={{ marginBottom: '40px' }}>
        <div className="info-badge">✨ Multimodal Analysis</div>
        <h1>Combined Speech + Text Analysis</h1>
        <p style={{ fontSize: '1.05rem', maxWidth: '600px' }}>
          Upload audio and enter transcript text for intelligent multimodal emotion recognition. Results combine both modalities using weighted fusion for comprehensive understanding.
        </p>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px', marginBottom: '40px' }}>
        {/* Input Section */}
        <section className="card">
          <h2 style={{ marginBottom: '20px' }}>📥 Upload Audio & Text</h2>
          <p style={{ marginBottom: '24px' }}>
            Provide both audio file and text to get intelligent fusion-based emotion prediction.
          </p>

          <form onSubmit={onSubmit}>
            {/* Audio Upload */}
            <div className="form-group">
              <label className="form-label">🎤 Audio File</label>
              <label className="file-dropzone" style={{ margin: 0 }}>
                <input 
                  type="file" 
                  accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.aac" 
                  onChange={handleFileChange} 
                />
                <div className="dropzone-content">
                  <div style={{ fontSize: '2rem', marginBottom: '8px' }}>🎵</div>
                  <span className="dropzone-title">Click or drag to upload audio</span>
                  <span className="dropzone-subtitle">WAV, MP3, M4A, OGG, FLAC, AAC</span>
                </div>
              </label>
            </div>

            {file && (
              <div style={{
                background: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid var(--success)',
                borderRadius: '10px',
                padding: '12px 16px',
                marginBottom: '20px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <span style={{ fontSize: '1.2rem' }}>✓</span>
                <div>
                  <p style={{ margin: 0, fontWeight: '600', color: 'var(--success)' }}>Audio file selected</p>
                  <p style={{ margin: '4px 0 0', fontSize: '0.9rem' }}>{fileLabel}</p>
                </div>
              </div>
            )}

            {/* Text Input */}
            <div className="form-group">
              <label className="form-label">📝 Transcript Text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                maxLength={5000}
                placeholder="Type or paste the transcript/dialogue associated with the audio"
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
                  minHeight: '140px',
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
              disabled={loading || !file || !text.trim()}
              style={{ width: '100%' }}
            >
              {loading ? '⏳ Analyzing...' : '🚀 Analyze Multimodal'}
            </button>
          </form>
        </section>

        {/* Results Section */}
        <section className="card">
          <h2 style={{ marginBottom: '20px' }}>📊 Fusion Results</h2>
          
          {loading && (
            <div className="empty-state">
              <div className="loading-spinner"></div>
              <p>Running speech & text models, fusing results...</p>
            </div>
          )}

          {!loading && result && (
            <div style={{ display: 'grid', gap: '20px' }}>
              <div>
                <p style={{ fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>🎯 Final Fused Prediction</p>
                <ResultCard 
                  title="Final Prediction" 
                  payload={{ 
                    predicted_emotion: result.final_prediction, 
                    confidence: result.confidence, 
                    all_probabilities: result.fusion_details.fused_probabilities 
                  }} 
                />
              </div>
            </div>
          )}

          {!loading && !result && (
            <div className="empty-state">
              <p style={{ fontSize: '1.05rem', color: 'var(--text-muted)' }}>
                👆 Upload audio and enter text for multimodal analysis
              </p>
            </div>
          )}
        </section>
      </div>

      {/* Individual Predictions */}
      {result && !loading && (
        <section style={{ marginTop: '40px' }}>
          <h2 style={{ marginBottom: '24px' }}>Individual Model Predictions</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
            <div>
              <p style={{ fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>🎤 Speech Model</p>
              <ResultCard 
                title="Speech Prediction" 
                fileName={file?.name}
                payload={{ 
                  predicted_emotion: result.speech_prediction, 
                  confidence: null, 
                  all_probabilities: result.speech_probabilities 
                }} 
              />
            </div>
            <div>
              <p style={{ fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>📝 Text Model</p>
              <ResultCard 
                title="Text Prediction" 
                payload={{ 
                  predicted_emotion: result.text_prediction, 
                  confidence: null, 
                  all_probabilities: result.text_probabilities 
                }} 
              />
            </div>
          </div>
        </section>
      )}

      {/* Fusion Details */}
      {result && !loading && (
        <section style={{ marginTop: '40px' }}>
          <h2 style={{ marginBottom: '24px' }}>⚙️ Fusion Configuration</h2>
          <div className="card">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
              <div>
                <p style={{ fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '8px' }}>Model Weights</p>
                <div style={{ display: 'grid', gap: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Speech</span>
                    <strong style={{ color: 'var(--emotion-angry)' }}>{(result.fusion_details.weights.speech * 100).toFixed(0)}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Text</span>
                    <strong style={{ color: 'var(--emotion-happy)' }}>{(result.fusion_details.weights.text * 100).toFixed(0)}%</strong>
                  </div>
                </div>
              </div>
              <div>
                <p style={{ fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '8px' }}>Emotion Labels</p>
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  {result.fusion_details.used_labels.join(', ')}
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Info Cards */}
      <section style={{ marginTop: '40px' }}>
        <h2 style={{ marginBottom: '24px', textAlign: 'center' }}>How Multimodal Fusion Works</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>🎤</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Speech Analysis</h3>
            <p style={{ fontSize: '0.9rem' }}>Deep learning model extracts acoustic features and predicts emotion with high accuracy</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📝</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Text Analysis</h3>
            <p style={{ fontSize: '0.9rem' }}>NLP model extracts semantic features and classifies emotional sentiment</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>🔗</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Intelligent Fusion</h3>
            <p style={{ fontSize: '0.9rem' }}>Weighted averaging combines both predictions for robust emotion recognition</p>
          </div>
        </div>
      </section>
    </div>
  )
}

