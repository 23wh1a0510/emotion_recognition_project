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

## Train Text Model

From repository root:

python models/text_pipeline/train.py --out_dir models/text_pipeline/results_test

Generated artifacts include:
- models/text_pipeline/results_test/text_model.joblib
- models/text_pipeline/results_test/training_log.csv
- models/text_pipeline/results_test/classification_report.txt
- models/text_pipeline/results_test/confusion_matrix.png
- models/text_pipeline/results_test/sanity_predictions.csv

## Run Full Stack

Backend:

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Frontend (production build):

cd frontend
npm run build
npx serve -s build -l 3000

Open http://localhost:3000 and test Speech or Text pages.
