import os
import glob
import requests
import json
import time

BASE_URL = os.environ.get('API_BASE','http://127.0.0.1:8001')
ENDPOINT = BASE_URL + '/api/multimodal/predict'

# find sample audio files by emotion if available
base = 'data/TESS'
emotions = ['angry','happy','neutral','sad','surprise','fear','disgust']
emap = {}
for emo in emotions:
    files = glob.glob(os.path.join(base,'**','*'+emo+'*.wav'), recursive=True)
    if files:
        emap[emo] = files[0]

fallback = 'backend/tmp/debug_sample.wav'
if not os.path.exists(fallback):
    # try some other fallback
    other = glob.glob('**/*.wav', recursive=True)
    fallback = other[0] if other else None

cases = [
    # matching
    (emap.get('angry', fallback), 'why did you do this', 'matching: angry/angry'),
    (emap.get('happy', fallback), 'I am so happy today', 'matching: happy/happy'),
    (emap.get('neutral', fallback), 'I am feeling okay', 'matching: neutral/neutral'),
    # conflicting
    (emap.get('angry', fallback), 'I am so happy today', 'conflict: angry(speech)/happy(text)'),
    (emap.get('happy', fallback), 'I feel terrible and sad', 'conflict: happy(speech)/sad(text)'),
    # neutral/empty
    (emap.get('neutral', fallback), '', 'neutral speech + empty text'),
]

results = []

for idx,(afile, text, desc) in enumerate(cases,1):
    print('='*80)
    print(f'Case {idx}: {desc}')
    if not afile:
        print('No audio file found; skipping case.')
        continue
    print('Using audio:', afile)
    try:
        with open(afile,'rb') as f:
            files = {'file': ('a.wav', f, 'audio/wav')}
            data = {'text': text}
            t0 = time.time()
            r = requests.post(ENDPOINT, files=files, data=data, timeout=30)
            elapsed = time.time() - t0
        print('HTTP', r.status_code)
        try:
            obj = r.json()
        except Exception:
            print('Non-JSON response:')
            print(r.text)
            obj = {'raw_text': r.text}
        # print concise summary if fields exist
        summary = {
            'case': desc,
            'audio': afile,
            'text': text,
            'http_status': r.status_code,
            'elapsed_s': round(elapsed,3),
            'response': obj,
        }
        results.append(summary)
        print('Speech prediction:', obj.get('speech_prediction') if isinstance(obj, dict) else 'n/a')
        print('Text prediction:', obj.get('text_prediction') if isinstance(obj, dict) else 'n/a')
        print('Fused prediction:', obj.get('final_prediction') if isinstance(obj, dict) else 'n/a')
        print('Full response JSON:')
        print(json.dumps(obj, indent=2))
    except Exception as e:
        print('Error during request:', e)

# save results
os.makedirs('results', exist_ok=True)
fn = 'results/multimodal_test_results.json'
with open(fn,'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=2)

print('\nSaved results to', fn)
print('Done.')
