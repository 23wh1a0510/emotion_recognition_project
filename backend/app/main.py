"""FastAPI backend entrypoint.

This module configures logging, CORS and mounts the API routers. It also creates
a temporary directory on startup where uploaded files will be stored before
they are processed by ML code (not implemented yet).

The API itself is defined in `app.routes`.
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routes


LOG = logging.getLogger("mer_backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="MER Backend")

# Allow frontend dev server origins (React default port 3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Create temp folder and log startup info."""
    tmp_dir = Path(__file__).resolve().parents[2] / "backend" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    app.state.tmp_dir = tmp_dir
    LOG.info(f"Startup complete — tmp_dir={tmp_dir}")

    # Attempt to load ML model if available. Paths can be overridden via env vars.
    try:
        from .model import MERModel
        # prefer common result dirs from training; allow both `results_test` and `results`
        candidate_dirs = [Path('models/speech_pipeline/results_test'), Path('models/speech_pipeline/results')]
        model_path = None
        le_path = None
        config_path = None
        for d in candidate_dirs:
            if d.exists():
                p_model = d / 'best_model.h5'
                p_le = d / 'label_encoder.joblib'
                p_cfg = d / 'config.json'
                if p_model.exists() and p_le.exists() and p_cfg.exists():
                    model_path = p_model
                    le_path = p_le
                    config_path = p_cfg
                    break
        # fallback to environment-specified paths if provided
        import os
        mp = os.getenv('MER_MODEL_PATH')
        lp = os.getenv('MER_LABEL_ENCODER_PATH')
        cp = os.getenv('MER_CONFIG_PATH')
        if mp:
            model_path = Path(mp)
        if lp:
            le_path = Path(lp)
        if cp:
            config_path = Path(cp)

        # allow overrides from environment
        import os
        mp = os.getenv('MER_MODEL_PATH')
        lp = os.getenv('MER_LABEL_ENCODER_PATH')
        cp = os.getenv('MER_CONFIG_PATH')
        if mp:
            model_path = Path(mp)
        if lp:
            le_path = Path(lp)
        if cp:
            config_path = Path(cp)

        if model_path.exists() and le_path.exists() and config_path.exists():
            app.state.mer_model = MERModel(str(model_path), str(le_path), str(config_path))
            app.state.mer_model.load()
            LOG.info("Loaded MER model into app.state.mer_model")
        else:
            LOG.warning("MER model files not found at default locations; speech inference will return 501 until model is available")
            app.state.mer_model = None
    except Exception:
        LOG.exception("Failed to initialize MER model on startup")
        app.state.mer_model = None

    try:
        from .model import TextEmotionModel

        candidate_artifacts = [
            Path('models/text_pipeline/results_test/text_model.joblib'),
            Path('models/text_pipeline/results/text_model.joblib'),
        ]
        artifact_path = None
        for p in candidate_artifacts:
            if p.exists():
                artifact_path = p
                break

        import os
        text_artifact_override = os.getenv('MER_TEXT_MODEL_ARTIFACT')
        if text_artifact_override:
            artifact_path = Path(text_artifact_override)

        if artifact_path and artifact_path.exists():
            app.state.text_model = TextEmotionModel(str(artifact_path))
            app.state.text_model.load()
            LOG.info("Loaded text emotion model into app.state.text_model")
        else:
            LOG.warning("Text model artifact not found at default locations; text inference will return 501 until model is available")
            app.state.text_model = None
    except Exception:
        LOG.exception("Failed to initialize text model on startup")
        app.state.text_model = None


app.include_router(routes.router, prefix="/api")


@app.get("/")
def root():
    LOG.info("Health check requested at /")
    return {"status": "ok", "message": "MER backend running. API mounted at /api"}
