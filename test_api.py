import requests
import json

API_URL = "http://localhost:8000"

def test_upload():
    print("--- 1. Testing Upload ---")
    url = f"{API_URL}/embeddings"
    
    # Replace these with real files on your disk
    files = [
        ('files', ('dog_bark.wav', open('dog_bark.wav', 'rb'), 'audio/wav')),
        ('files', ('piano.wav', open('piano.wav', 'rb'), 'audio/wav'))
    ]
    
    response = requests.post(url, files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_search():
    print("\n--- 2. Testing Search ---")
    url = f"{API_URL}/search"
    
    # We search using the dog bark. 
    # The API should return 'dog_bark.wav' as the #1 result with high similarity.
    files = {
        'file': ('query_dog.wav', open('dog_bark.wav', 'rb'), 'audio/wav')
    }
    data = {'top_k': 3}
    
    response = requests.post(url, files=files, data=data)
    
    print(f"Status: {response.status_code}")
    results = response.json()
    
    print("Top Matches:")
    for item in results:
        print(f"- {item['filename']} (Score: {item['similarity_score']:.4f})")

if __name__ == "__main__":
    # Ensure you have the audio files before running!
    try:
        test_upload()
        test_search()
    except FileNotFoundError:
        print("Error: Please put 'dog_bark.wav' and 'piano.wav' in this folder to run tests.")