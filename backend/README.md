# Backend (FastAPI) — MER Project

This folder contains the FastAPI scaffold for MER. Endpoints are placeholders returning 501 Not Implemented until models are added.

How to install and run (backend):

1. Create a Python virtual environment (`python -m venv .venv`) and activate it.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Run the server with uvicorn:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API root is `http://localhost:8000/` and API endpoints are under `/api`.

Available endpoints (placeholders returning a simple JSON for now):

- `POST /api/speech/predict`  — form-data with **file** (audio)
- `POST /api/text/predict`    — form-data with **text** (string)
- `POST /api/multimodal/predict` — form-data with **file** + **text**

All three endpoints validate inputs, save uploaded audio to `backend/tmp/`, and
return JSON `{ "success": true, "emotion": "processing_not_added_yet" }`.

Speech inference (real model):
- If a trained model is available under `models/speech_pipeline/results` or `models/speech_pipeline/results_test`, the backend will load it on startup and the `POST /api/speech/predict` endpoint will perform real preprocessing and model inference.

Expected POST body for speech inference:
- form-data with key `file` (audio file)

Example response (real inference):
```
{
	"success": true,
	"predicted_emotion": "happy",
	"confidence": 0.94,
	"all_probabilities": { "happy": 0.94, "sad": 0.02, ... }
}
```

If you trained the model into a custom folder, set environment variables before starting uvicorn:
- `MER_MODEL_PATH` — path to the saved Keras model (best_model.h5)
- `MER_LABEL_ENCODER_PATH` — path to `label_encoder.joblib`
- `MER_CONFIG_PATH` — path to `config.json`

Start the backend with uvicorn as usual. On startup the server will attempt to load the model and log status.
