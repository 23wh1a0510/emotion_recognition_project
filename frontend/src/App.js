import React from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Home from './pages/Home'
import Speech from './pages/Speech'
import Text from './pages/Text'
import Multimodal from './pages/Multimodal'

// App defines top-level navigation and routes for all pages required by the project.
export default function App() {
  const location = useLocation()

  return (
    <div>
      <nav>
        <div>
          <Link 
            to="/" 
            style={{ 
              fontWeight: '700',
              fontSize: '1.2rem',
              background: 'linear-gradient(135deg, #4facfe 0%, #f093fb 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              marginRight: '40px'
            }}
          >
            🎵 MER
          </Link>
          <Link 
            to="/" 
            className={location.pathname === '/' ? 'active' : ''}
            style={{ textDecoration: 'none' }}
          >
            Home
          </Link>
          <Link 
            to="/speech" 
            className={location.pathname === '/speech' ? 'active' : ''}
            style={{ textDecoration: 'none' }}
          >
            🗣️ Speech
          </Link>
          <Link 
            to="/text" 
            className={location.pathname === '/text' ? 'active' : ''}
            style={{ textDecoration: 'none' }}
          >
            📝 Text
          </Link>
          <Link 
            to="/multimodal" 
            className={location.pathname === '/multimodal' ? 'active' : ''}
            style={{ textDecoration: 'none' }}
          >
            ✨ Multimodal
          </Link>
        </div>
      </nav>

      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/speech" element={<Speech />} />
          <Route path="/text" element={<Text />} />
          <Route path="/multimodal" element={<Multimodal />} />
        </Routes>
      </main>
    </div>
  )
}
