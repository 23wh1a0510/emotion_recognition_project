import React, { useMemo, useState } from 'react'
import api from '../api'
import ResultCard from '../components/ResultCard'

export default function Speech() {
  const [file, setFile] = useState(null)
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
    const selectedFile = e.target.files?.[0] || null
    setError('')
    setStatus('')
    setResult(null)

    if (!selectedFile) {
      setFile(null)
      return
    }

    if (!isValidAudio(selectedFile)) {
      setFile(null)
      setError('Please choose a valid audio file such as WAV, MP3, M4A, OGG, FLAC, or AAC.')
      return
    }

    setFile(selectedFile)
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)
    setStatus('')
    if (!file) {
      setError('Please select an audio file before submitting.')
      return
    }

    if (!isValidAudio(file)) {
      setError('Selected file is not a supported audio type.')
      return
    }

    setLoading(true)
    setStatus('Uploading audio and running speech inference...')
    try {
      const res = await api.postSpeech(file)
      setResult(res)
      setStatus('Prediction complete.')
    } catch (err) {
      setError(err.message || 'Unable to predict speech emotion right now.')
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="speech-page">
      {/* Hero Section */}
      <section style={{ marginBottom: '40px' }}>
        <div className="info-badge">🎤 Speech Analysis</div>
        <h1>Speech Emotion Recognition</h1>
        <p style={{ fontSize: '1.05rem', maxWidth: '600px' }}>
          Upload an audio file to analyze emotional tone using advanced deep learning. Get instant emotion prediction with confidence scores and probability distributions.
        </p>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px', marginBottom: '40px' }}>
        {/* Upload Section */}
        <section className="card">
          <h2 style={{ marginBottom: '20px' }}>📁 Select Audio</h2>
          <p style={{ marginBottom: '24px' }}>
            Choose a WAV, MP3, M4A, OGG, FLAC, or AAC file for emotion analysis.
          </p>

          <form onSubmit={onSubmit}>
            <div className="form-group">
              <label htmlFor="audio-input" className="form-label">Audio File</label>
              <label className="file-dropzone">
                <input 
                  id="audio-input"
                  type="file" 
                  accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.aac" 
                  onChange={handleFileChange} 
                />
                <div className="dropzone-content">
                  <div style={{ fontSize: '2rem', marginBottom: '8px' }}>🎵</div>
                  <span className="dropzone-title">Click or drag to upload</span>
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
                  <p style={{ margin: 0, fontWeight: '600', color: 'var(--success)' }}>File selected</p>
                  <p style={{ margin: '4px 0 0', fontSize: '0.9rem' }}>{fileLabel}</p>
                </div>
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
              disabled={loading}
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
              <p>Running deep learning inference on your audio...</p>
            </div>
          )}

          {!loading && result && <ResultCard fileName={file?.name} payload={result} />}
          
          {!loading && !result && (
            <div className="empty-state">
              <p style={{ fontSize: '1.05rem', color: 'var(--text-muted)' }}>
                👆 Upload an audio file to see emotion prediction results
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
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Upload Audio</h3>
            <p style={{ fontSize: '0.9rem' }}>Select a WAV, MP3, or other audio format file from your device</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>2️⃣</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Process</h3>
            <p style={{ fontSize: '0.9rem' }}>Deep learning model extracts features and runs real-time inference</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>3️⃣</div>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '8px' }}>Results</h3>
            <p style={{ fontSize: '0.9rem' }}>Get emotion prediction, confidence score, and probability breakdown</p>
          </div>
        </div>
      </section>
    </div>
  )
}
