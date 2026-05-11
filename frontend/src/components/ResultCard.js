import React from 'react'

function formatProbability(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function probabilityBars(probabilities) {
  return Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1])
}

const emotionEmojis = {
  angry: '😠',
  disgust: '🤮',
  fear: '😨',
  happy: '😊',
  neutral: '😐',
  sad: '😢',
  surprise: '😮'
}

const emotionColors = {
  angry: '#ff6b6b',
  disgust: '#c77dff',
  fear: '#a8dadc',
  happy: '#ffd93d',
  neutral: '#95e1d3',
  sad: '#6bcf7f',
  surprise: '#06ffa5'
}

export default function ResultCard({ title = 'Speech Prediction', fileName, payload }) {
  if (!payload) return null

  const bars = probabilityBars(payload.all_probabilities)
  const confidence = formatProbability(payload.confidence)
  const emotion = payload.predicted_emotion?.toLowerCase()
  const emotionColor = emotionColors[emotion] || '#667eea'
  const emotionEmoji = emotionEmojis[emotion] || '🎵'

  return (
    <div 
      className="result-card"
      style={{
        borderColor: emotionColor,
        background: `linear-gradient(135deg, rgba(${parseInt(emotionColor.slice(1,3),16)}, ${parseInt(emotionColor.slice(3,5),16)}, ${parseInt(emotionColor.slice(5,7),16)}, 0.08) 0%, rgba(102, 126, 234, 0.05) 100%)`,
      }}
    >
      <div className="result-card-header">
        <div style={{ flex: 1 }}>
          <span className="result-eyebrow">{title}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '8px' }}>
            <span style={{ fontSize: '2.2rem' }}>{emotionEmoji}</span>
            <h3 style={{ margin: 0, color: emotionColor, fontSize: '2rem' }}>
              {payload.predicted_emotion}
            </h3>
          </div>
        </div>
        <div 
          className="result-confidence"
          style={{
            background: `rgba(${parseInt(emotionColor.slice(1,3),16)}, ${parseInt(emotionColor.slice(3,5),16)}, ${parseInt(emotionColor.slice(5,7),16)}, 0.15)`,
            borderColor: emotionColor,
            border: `1px solid ${emotionColor}`
          }}
        >
          <span>Confidence</span>
          <strong style={{ color: emotionColor }}>{confidence}</strong>
        </div>
      </div>

      {fileName && (
        <div className="result-filename">
          <span>📁 Input</span>
          <strong>{fileName}</strong>
        </div>
      )}

      <div className="result-shapes">
        {payload.input_shape && <span>🔧 Input shape: {payload.input_shape.join(' × ')}</span>}
        {payload.feature_shape && <span>📊 Feature shape: {payload.feature_shape.join(' × ')}</span>}
        {payload.elapsed_seconds != null && <span>⚡ Latency: {Number(payload.elapsed_seconds).toFixed(2)}s</span>}
      </div>

      <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: `1px solid rgba(${parseInt(emotionColor.slice(1,3),16)}, ${parseInt(emotionColor.slice(3,5),16)}, ${parseInt(emotionColor.slice(5,7),16)}, 0.2)` }}>
        <p style={{ fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '16px' }}>
          Emotion Probabilities
        </p>
        <div className="probability-list">
          {bars.map(([label, value]) => {
            const labelLower = label.toLowerCase()
            const labelColor = emotionColors[labelLower] || '#667eea'
            const percentage = Math.max(0, Math.min(100, Number(value) * 100))
            
            return (
              <div className="probability-row" key={label}>
                <div className="probability-labels">
                  <span>{emotionEmojis[labelLower] || '🎵'} {label}</span>
                  <strong style={{ color: labelColor }}>{formatProbability(value)}</strong>
                </div>
                <div className="probability-bar-track" style={{ background: `rgba(${parseInt(labelColor.slice(1,3),16)}, ${parseInt(labelColor.slice(3,5),16)}, ${parseInt(labelColor.slice(5,7),16)}, 0.1)` }}>
                  <div 
                    className="probability-bar-fill" 
                    style={{ 
                      width: `${percentage}%`,
                      background: labelColor
                    }} 
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

