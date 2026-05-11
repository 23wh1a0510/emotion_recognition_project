#!/usr/bin/env python3
"""
Comprehensive multimodal emotion recognition testing.
Tests multiple cases: matching emotions, conflicting emotions, and neutral cases.
"""

import requests
import json
from pathlib import Path

API_BASE = "http://127.0.0.1:8000/api"
# Use available test audio samples from backend/tmp or TESS dataset
TEST_AUDIO_FILES = {
    "angry": "backend/tmp/angry.wav",
    "happy": "data/TESS/TESS Toronto emotional speech set data/OAF_happy/OAF_back_happy.wav",
    "sad": "data/TESS/TESS Toronto emotional speech set data/OAF_sad/OAF_back_sad.wav",
    "neutral": "data/TESS/TESS Toronto emotional speech set data/OAF_neutral/OAF_back_neutral.wav",
}

def run_test_case(case_name, audio_key, text_input):
    """Run a single test case and return results"""
    print(f"\n{'='*70}")
    print(f"TEST CASE: {case_name}")
    print(f"{'='*70}")
    
    audio_file = TEST_AUDIO_FILES.get(audio_key)
    if not audio_file or not Path(audio_file).exists():
        print(f"⚠ Warning: Audio file {audio_file} not found")
        return None
    
    with open(audio_file, 'rb') as f:
        files = {'file': f}
        data = {'text': text_input}
        
        try:
            resp = requests.post(f"{API_BASE}/multimodal/predict", files=files, data=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                
                # Display results
                print(f"\n📝 Text Input: '{text_input}'")
                print(f"\n🗣️  SPEECH PREDICTION (from {audio_key} audio):")
                print(f"   Emotion: {result['speech_prediction'].upper()}")
                print(f"   Confidence: {result['speech_probabilities'][result['speech_prediction']]*100:.2f}%")
                
                print(f"\n📄 TEXT PREDICTION:")
                print(f"   Emotion: {result['text_prediction'].upper()}")
                print(f"   Confidence: {result['text_probabilities'][result['text_prediction']]*100:.2f}%")
                
                print(f"\n🔀 FINAL FUSED PREDICTION:")
                print(f"   Emotion: {result['final_prediction'].upper()}")
                print(f"   Final Confidence: {result['confidence']*100:.2f}%")
                
                print(f"\n⚙️  FUSION DETAILS:")
                print(f"   Speech weight: {result['fusion_details']['weights']['speech']}")
                print(f"   Text weight: {result['fusion_details']['weights']['text']}")
                print(f"   Fused {result['final_prediction']} prob: {result['fusion_details']['fused_probabilities'][result['final_prediction']]*100:.2f}%")
                
                return result
            else:
                print(f"❌ Error: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE MULTIMODAL EMOTION RECOGNITION TEST")
    print("="*70)
    print(f"Backend API: {API_BASE}")
    print(f"Available test audio: {list(TEST_AUDIO_FILES.keys())}")
    
    # Test cases covering different scenarios
    test_cases = [
        ("Matching - Angry Audio + Angry Text", "angry", "I am so angry right now!"),
        ("Matching - Happy Audio + Happy Text", "happy", "This is amazing! I am so happy!"),
        ("Conflicting - Angry Audio + Happy Text", "angry", "I love this! Everything is wonderful!"),
        ("Conflicting - Happy Audio + Sad Text", "happy", "This is terrible. I feel so sad and broken."),
        ("Neutral - Neutral Audio + Neutral Text", "neutral", "The weather is nice today."),
        ("Conflicting - Sad Audio + Happy Text", "sad", "I am thrilled and excited about this!"),
    ]
    
    results = []
    for case_name, audio_key, text_input in test_cases:
        result = run_test_case(case_name, audio_key, text_input)
        if result:
            results.append({
                "case": case_name,
                "audio_type": audio_key,
                "text": text_input,
                "result": result
            })
    
    # Summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total cases run: {len(results)}")
    
    print("\nRESULTS TABLE:")
    print(f"{'Case':<50} {'Audio':<8} {'Text':<8} {'Speech':<8} {'Final':<8} {'Conf':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r['case']:<50} {r['audio_type']:<8} {r['result']['text_prediction']:<8} {r['result']['speech_prediction']:<8} {r['result']['final_prediction']:<8} {r['result']['confidence']*100:>6.2f}%")
    
    # Save detailed results to JSON
    output_file = Path("results/comprehensive_test_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
