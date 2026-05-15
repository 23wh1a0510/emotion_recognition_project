import traceback
try:
    from transformers import pipeline
    print('transformers imported')
    p = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli', device=-1)
    print('pipeline created', type(p))
except Exception as e:
    print('ERROR initializing pipeline')
    traceback.print_exc()
