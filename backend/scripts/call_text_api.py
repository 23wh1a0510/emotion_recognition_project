import urllib.parse
from urllib import request
url = 'http://127.0.0.1:8000/api/text/predict'
form = urllib.parse.urlencode({'text': 'I am absolutely thrilled and delighted with everything'}).encode('utf8')
req = request.Request(url, data=form, headers={'Content-Type':'application/x-www-form-urlencoded'})
with request.urlopen(req, timeout=30) as resp:
    print(resp.read().decode('utf8'))
