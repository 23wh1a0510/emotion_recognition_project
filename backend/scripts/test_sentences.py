import urllib.parse
from urllib import request
import json

url = 'http://127.0.0.1:8000/api/text/predict'
inputs = [
    "I'm fine.",
    "I can't do this anymore.",
    "This is the best day of my life!",
    "Why would you do that to me?",
]

for t in inputs:
    form = urllib.parse.urlencode({'text': t}).encode('utf8')
    req = request.Request(url, data=form, headers={'Content-Type':'application/x-www-form-urlencoded'})
    try:
        with request.urlopen(req, timeout=30) as resp:
            print('INPUT:', t)
            print(json.dumps(json.loads(resp.read().decode('utf8')), indent=2))
    except Exception as e:
        print('ERROR for input:', t)
        print(e)
    print('\n' + '-'*60 + '\n')
