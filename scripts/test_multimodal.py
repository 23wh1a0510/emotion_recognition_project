import requests
from pathlib import Path
base = Path('data/TESS/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav')
if not base.exists():
    base = Path('backend/tmp/debug_sample.wav')
with base.open('rb') as f:
    try:
        r = requests.post('http://127.0.0.1:8000/api/multimodal/predict', files={'file': ('angry.wav', f, 'audio/wav')}, data={'text':'I am so happy today'}, timeout=30)
        print('CODE', r.status_code)
        print(r.text)
    except Exception as e:
        print('ERR', e)
