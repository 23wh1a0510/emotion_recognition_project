"""Test script for text pipeline.

This script performs a quick artifact check for the text pipeline and
prints which expected outputs are available.
"""
import os

def main():
    print('Text pipeline test: checking for artifacts')
    base = os.path.join('models', 'text_pipeline', 'results_test')
    expected = ['text_model.joblib', 'classification_report.txt', 'confusion_matrix.png']
    for f in expected:
        path = os.path.join(base, f)
        print(f, '->', 'FOUND' if os.path.exists(path) else 'MISSING')

if __name__ == '__main__':
    main()
