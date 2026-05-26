# Multimodal Emotion Recognition (MER)

This repository now includes real emotion recognition pipelines for:
- Speech emotion recognition (TensorFlow Conv1D + BiLSTM)
- Text emotion recognition (TF-IDF + Logistic Regression)

Multimodal fusion is intentionally still pending.

## Backend API

- Speech: POST /api/speech/predict
- Text: POST /api/text/predict
- Multimodal: placeholder only

Speech and text endpoints return:
- predicted_emotion
- confidence
- all_probabilities

## Project Structure & Training

The repository follows the deliverable structure used for submission. Key folders:

```
models/
	speech_pipeline/
	text_pipeline/
	fusion_pipeline/
Results/
	accuracy_tables/
	plots/
```

Train/test entrypoints for each pipeline (placeholders) exist under `models/*_pipeline/`.

### Example: train or test a pipeline

From repository root:

```powershell
python models\text_pipeline\train.py --out_dir models\text_pipeline\results_test
python models\text_pipeline\test.py
```

Artifacts produced by training (if you run training) will be written into each
pipeline's `results_test` folder. Existing evaluation reports are available in
`Results/accuracy_tables` and plots are referenced from `Results/plots/README.md`.
## Run Full Stack

Backend:

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Frontend (production build):

cd frontend //

npm run build //

npx serve -s build -l 3000

Open http://localhost:3000 and test Speech or Text pages.

Run Evaluations

From the project root:

python evaluate_speech.py //
python evaluate_text.py  //
python evaluate_transformer.py //
python evaluate_multimodal.py
