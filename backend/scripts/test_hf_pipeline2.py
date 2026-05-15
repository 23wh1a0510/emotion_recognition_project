import os
from pathlib import Path
import traceback
project_root = Path(__file__).resolve().parents[2]
cache = project_root / '.hf_cache'
cache.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(cache)
os.environ['TRANSFORMERS_CACHE'] = str(cache)
os.environ['HF_DATASETS_CACHE'] = str(cache)
print('HF cache set to', cache)
try:
    from transformers import pipeline
    print('transformers imported')
    p = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli', device=-1)
    print('pipeline created', type(p))
except Exception:
    print('ERROR initializing pipeline')
    traceback.print_exc()
