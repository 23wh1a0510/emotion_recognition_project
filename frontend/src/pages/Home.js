import React from 'react'
import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <h1>🎵 Emotion Recognition</h1>
        <p style={{ fontSize: '1.1rem', marginTop: '16px' }}>
          Advanced multimodal emotion recognition combining speech analysis, text understanding, and intelligent fusion
        </p>
        
        <div className="cta-buttons">
          <Link to="/speech" className="btn btn-primary" style={{ textDecoration: 'none' }}>
            🗣️ Analyze Speech
          </Link>
          <Link to="/text" className="btn btn-secondary" style={{ textDecoration: 'none' }}>
            📝 Analyze Text
          </Link>
          <Link to="/multimodal" className="btn btn-primary" style={{ textDecoration: 'none' }}>
            ✨ Combined Analysis
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section>
        <h2 style={{ textAlign: 'center', marginBottom: '40px' }}>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎤</div>
            <h3>Speech Analysis</h3>
            <p>
              Advanced deep learning model analyzes emotional tone from audio files with confidence scoring across 7 emotion categories
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">💬</div>
            <h3>Text Analysis</h3>
            <p>
              NLP-powered emotion detection from written text using machine learning feature extraction and classification
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>Multimodal Fusion</h3>
            <p>
              Intelligent combination of speech and text predictions using weighted fusion for comprehensive emotion understanding
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Detailed Results</h3>
            <p>
              Probability distributions across all emotions with confidence metrics and probability visualization
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Real-time Processing</h3>
            <p>
              Instant emotion recognition with minimal latency for interactive analysis and rapid feedback
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🎨</div>
            <h3>Modern Interface</h3>
            <p>
              Professional, intuitive UI with smooth animations and glassmorphism design for optimal user experience
            </p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section style={{ marginTop: '60px' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '40px' }}>How It Works</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px' }}>
          <div className="card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--primary)' }}>1️⃣ Upload</h3>
            <p>
              Choose an audio file (for speech analysis) or enter text (for text analysis) or both (for combined multimodal analysis)
            </p>
          </div>

          <div className="card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--accent-light)' }}>2️⃣ Process</h3>
            <p>
              Our ML models extract features and run real-time inference using Keras (speech) and scikit-learn (text) pipelines
            </p>
          </div>

          <div className="card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--accent)' }}>3️⃣ Results</h3>
            <p>
              Get instant emotion predictions with confidence scores and probability distributions for all 7 emotion categories
            </p>
          </div>
        </div>
      </section>

      {/* Emotions Section */}
      <section style={{ marginTop: '60px' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '40px' }}>Recognized Emotions</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '16px' }}>
          {[
            { emoji: '😠', label: 'Angry', color: 'var(--emotion-angry)' },
            { emoji: '😐', label: 'Disgust', color: 'var(--emotion-disgust)' },
            { emoji: '😨', label: 'Fear', color: 'var(--emotion-fear)' },
            { emoji: '😊', label: 'Happy', color: 'var(--emotion-happy)' },
            { emoji: '😐', label: 'Neutral', color: 'var(--emotion-neutral)' },
            { emoji: '😢', label: 'Sad', color: 'var(--emotion-sad)' },
            { emoji: '😮', label: 'Surprise', color: 'var(--emotion-surprise)' },
          ].map(emotion => (
            <div 
              key={emotion.label}
              className="card"
              style={{ 
                textAlign: 'center',
                borderColor: emotion.color,
                background: `rgba(${emotion.color}, 0.05)`,
              }}
            >
              <div style={{ fontSize: '2rem', marginBottom: '8px' }}>{emotion.emoji}</div>
              <p style={{ fontWeight: '600', color: emotion.color, margin: 0 }}>{emotion.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Footer */}
      <section style={{ marginTop: '60px', textAlign: 'center' }}>
        <div className="info-badge">✨ Production-Ready Models</div>
        <h2>Ready to Analyze?</h2>
        <p style={{ fontSize: '1.05rem', marginBottom: '30px' }}>
          Start with speech, text, or multimodal emotion recognition using our advanced ML models
        </p>
        <div className="cta-buttons">
          <Link to="/speech" className="btn btn-primary" style={{ textDecoration: 'none' }}>
            Get Started
          </Link>
          <Link to="/multimodal" className="btn btn-secondary" style={{ textDecoration: 'none' }}>
            Try Multimodal
          </Link>
        </div>
      </section>
    </div>
  )
}

