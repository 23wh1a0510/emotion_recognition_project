# Frontend (React) — MER Project

This frontend connects the Speech page to the real FastAPI speech inference API.

## Run the full app

1. Start the backend from the repository root:

```bash
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

2. Start the frontend from the `frontend/` folder:

```bash
npm install
npm start
```

The React dev server runs on port 3000 and proxies `/api/*` to the FastAPI backend on `http://localhost:8000`.

## Speech prediction test

1. Open the Speech page in the browser.
2. Upload a WAV or MP3 file.
3. Wait for the loading state to finish.
4. Confirm the UI shows the predicted emotion, confidence, all probabilities, and uploaded filename.

The page now uses only real backend output from `POST /api/speech/predict`.
