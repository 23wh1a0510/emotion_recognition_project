"""API routes for MER backend.

This module implements endpoints for speech, text and multimodal inputs.
Endpoints perform basic validation, save uploaded audio to a temporary folder
and return a dummy JSON response indicating that ML inference is not yet
implemented. All responses follow a simple JSON structure so the frontend can
be wired up before ML code is added.

Endpoints provided (placeholders raising 501 until ML code is implemented):
- GET  /api/health
- POST /api/speech/predict       (file upload)
- POST /api/text/predict         (form: text)
- POST /api/multimodal/predict   (file + text)

These endpoints are wired for speech inference and placeholder responses for the
text and multimodal paths until those models are added.
"""

import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

from . import utils

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/speech/predict")
async def speech_predict(request: Request, file: UploadFile = File(...)):
    """Accept an audio upload, validate it, save it, and run speech inference."""
    request_started = time.perf_counter()
    try:
        utils.LOG.info("Speech predict request received; filename=%s content_type=%s", file.filename, file.content_type)

        # Validate basic file properties
        utils.validate_audio_file(file)

        # Save to backend tmp directory created at startup
        tmp_dir: str = str(request.app.state.tmp_dir)
        save_started = time.perf_counter()
        saved_path = await utils.save_upload_file_tmp(file, tmp_dir)
        utils.LOG.info("Audio file saved in %.3fs to %s", time.perf_counter() - save_started, saved_path)

        # Ensure model is loaded
        mer_model = getattr(request.app.state, 'mer_model', None)
        if mer_model is None:
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech model not available on server")

        # Run inference
        result = mer_model.infer_audio(saved_path)
        utils.LOG.info("Speech prediction complete for %s: %s", file.filename, result)

        resp = {
            "success": True,
            "predicted_emotion": result['predicted_emotion'],
            "confidence": result['confidence'],
            "all_probabilities": result['all_probabilities'],
            "feature_shape": result.get('feature_shape'),
            "input_shape": result.get('input_shape'),
            "elapsed_seconds": result.get('elapsed_seconds'),
        }
        utils.LOG.info("Response returned in %.3fs for filename=%s", time.perf_counter() - request_started, file.filename)
        return JSONResponse(resp)

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - generic server error
        utils.LOG.exception("Unhandled error in speech_predict")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/text/predict")
async def text_predict(request: Request, text: str = Form(...)):
    """Accept raw transcript text for future prediction.

    Performs basic validation (non-empty, reasonable length) and returns a
    dummy JSON response.
    """
    try:
        text = text.strip()
        if not text:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Text cannot be empty")
        if len(text) > 5000:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Text too long")

        utils.LOG.info("Text predict request received; length=%d", len(text))

        text_model = getattr(request.app.state, 'text_model', None)
        if text_model is None:
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Text model not available on server")

        result = text_model.infer_text(text)
        utils.LOG.info("Text prediction complete: %s", result)

        return JSONResponse(
            {
                "success": True,
                "predicted_emotion": result['predicted_emotion'],
                "confidence": result['confidence'],
                "all_probabilities": result['all_probabilities'],
                "cleaned_text": result.get('cleaned_text'),
                "token_count": result.get('token_count'),
                "elapsed_seconds": result.get('elapsed_seconds'),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        utils.LOG.exception("Unhandled error in text_predict")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/multimodal/predict")
async def multimodal_predict(request: Request, file: UploadFile = File(...), text: str = Form(...)):
    """Accept both audio and text, run speech and text inference, fuse results, and return final prediction.

    Fusion strategy: weighted averaging of probability vectors from speech and text models.
    Default weights: speech=0.6, text=0.4 (configurable later).
    """
    try:
        # basic text validation
        text = text.strip()
        if not text:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Text cannot be empty")

        utils.validate_audio_file(file)

        # Save audio
        tmp_dir: str = str(request.app.state.tmp_dir)
        saved_path = await utils.save_upload_file_tmp(file, tmp_dir)
        utils.LOG.info("Saved multimodal audio to %s; text_len=%d", saved_path, len(text))

        # Ensure models available
        mer_model = getattr(request.app.state, 'mer_model', None)
        text_model = getattr(request.app.state, 'text_model', None)
        if mer_model is None:
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech model not available on server")
        if text_model is None:
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Text model not available on server")

        # Run speech inference (returns same format as speech endpoint)
        speech_result = mer_model.infer_audio(saved_path)

        # Run text inference
        text_result = text_model.infer_text(text)

        # Prepare probability maps aligned on label set
        speech_probs = speech_result.get('all_probabilities', {})
        text_probs = text_result.get('all_probabilities', {})

        # Determine label set: union of both; prefer labels present in speech label encoder
        speech_labels = list(mer_model.label_encoder.classes_)
        text_labels = list(text_model.label_encoder.classes_)
        labels = list(dict.fromkeys(speech_labels + text_labels))

        # Default fusion weights (can be tuned)
        w_speech = float(request.query_params.get('w_speech', 0.6))
        w_text = float(request.query_params.get('w_text', 0.4))

        fused = {}
        for lbl in labels:
            p_s = float(speech_probs.get(lbl, 0.0))
            p_t = float(text_probs.get(lbl, 0.0))
            fused[lbl] = w_speech * p_s + w_text * p_t

        # Normalize fused probs to sum to 1 (in case weights don't sum to 1)
        total = sum(fused.values())
        if total > 0:
            fused = {k: float(v / total) for k, v in fused.items()}

        # Final prediction
        final_label = max(fused.items(), key=lambda x: x[1])[0]
        final_conf = float(fused[final_label])

        resp = {
            "success": True,
            "speech_prediction": speech_result.get('predicted_emotion'),
            "text_prediction": text_result.get('predicted_emotion'),
            "final_prediction": final_label,
            "confidence": final_conf,
            "speech_probabilities": speech_probs,
            "text_probabilities": text_probs,
            "fusion_details": {
                "weights": {"speech": w_speech, "text": w_text},
                "used_labels": labels,
                "fused_probabilities": fused,
            },
        }

        utils.LOG.info("Multimodal prediction completed; final=%s conf=%.4f", final_label, final_conf)
        return JSONResponse(resp)

    except HTTPException:
        raise
    except Exception as exc:
        utils.LOG.exception("Unhandled error in multimodal_predict")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

