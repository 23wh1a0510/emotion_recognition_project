// Reusable API helper functions for frontend-backend communication.
// Development uses CRA proxy (relative `/api/...`), while production build
// talks directly to the local FastAPI server unless overridden.

const RAW_API_BASE =
  process.env.REACT_APP_API_BASE_URL ||
  (process.env.NODE_ENV === 'production' ? 'http://127.0.0.1:8000' : '')

const API_BASE = RAW_API_BASE.replace(/\/$/, '')

function apiUrl(path) {
  return `${API_BASE}${path}`
}

async function _handleResponse(res) {
  const contentType = res.headers.get('content-type') || ''
  let payload = null
  if (contentType.includes('application/json')) {
    payload = await res.json()
  } else {
    payload = await res.text()
  }

  if (!res.ok) {
    const message = payload && payload.detail ? payload.detail : res.statusText
    throw new Error(message || 'Server error')
  }
  return payload
}

export async function postSpeech(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(apiUrl('/api/speech/predict'), { method: 'POST', body: form })
  return _handleResponse(res)
}

export async function postText(text) {
  const form = new FormData()
  form.append('text', text)
  const res = await fetch(apiUrl('/api/text/predict'), { method: 'POST', body: form })
  return _handleResponse(res)
}

export async function postMultimodal(file, text) {
  const form = new FormData()
  form.append('file', file)
  form.append('text', text)
  const res = await fetch(apiUrl('/api/multimodal/predict'), { method: 'POST', body: form })
  return _handleResponse(res)
}

export default { postSpeech, postText, postMultimodal }
