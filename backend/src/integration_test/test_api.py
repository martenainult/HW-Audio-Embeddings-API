import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://localhost:8000"

def test_upload_all_in_folder():
    print(f"--- 1. Testing Batch Upload from: {BASE_DIR} ---")
    url = f"{API_URL}/embeddings"
    audio_extensions = {".wav", ".mp3"}
    audio_files = [f for f in BASE_DIR.iterdir() if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print("No audio files found!")
        return

    files_to_upload = []
    opened_files = []
    for file_path in audio_files:
        f = open(file_path, 'rb')
        opened_files.append(f)
        files_to_upload.append(('files', (file_path.name, f, 'audio/wav')))

    try:
        response = requests.post(url, files=files_to_upload)
        print(f"Upload Status: {response.status_code}")
    finally:
        for f in opened_files:
            f.close()

def test_invalid_file_rejection():
    """
    Test Case: Verifies that uploading a non-audio file results in an error 
    status rather than a server crash.
    """
    print("\n--- Testing Security: Invalid File Rejection ---")
    url = f"{API_URL}/embeddings"
    
    # Create a fake 'audio' file that is actually just text
    fake_audio_content = b"This is not audio data, it is just a text string."
    files = [
        ('files', ('malicious_fake.wav', fake_audio_content, 'audio/wav'))
    ]
    
    try:
        response = requests.post(url, files=files)
        data = response.json()
        
        # We expect a 200 OK because the API processes the list, 
        # but the specific file result should be 'error'
        file_result = data.get('malicious_fake.wav', {})
        
        if file_result.get('status') == 'error':
            print("[✓] PASSED: API correctly identified and rejected invalid audio data.")
            print(f"    Error Message: {file_result.get('message')}")
        else:
            print("[✗] FAILED: API should have returned an error status for invalid data.")
            
    except Exception as e:
        print(f"[✗] FAILED: The server likely crashed or timed out: {e}")

def test_list_all_files():
    """New Method: Fetches and prints the first 25 records from the DB."""
    print("\n--- 2. Listing Collection Contents ---")
    url = f"{API_URL}/embeddings"
    
    response = requests.get(url)
    if response.status_code == 200:
        records = response.json()
        total = len(records)
        print(f"Total records in database: {total}")
        
        # Logic: Show only the first 25
        display_count = 25
        for idx, item in enumerate(records[:display_count], 1):
            print(f"{idx}. {item['filename']} (ID: {item['id'][:8]}...)")
            
        if total > display_count:
            print(f"... and {total - display_count} more records.")
    else:
        print(f"Error listing files: {response.text}")

def test_search_with_first_found():
    print("\n--- 3. Testing Search ---")
    url = f"{API_URL}/search"
    try:
        query_file = next(BASE_DIR.glob("test_*.wav"))
    except StopIteration:
        return


    with open(query_file, 'rb') as f:
        files = {'file': (query_file.name, f, 'audio/wav')}
        response = requests.post(url, files=files, data={'top_k': 5})
    
    print(f"Search Status: {response.status_code}")
    print(f"Search results for '{query_file.name}':\n")
    
    for item in response.json():
        print(f"- {item['filename']} (Score: {item['similarity_score']:.4f})")
    
def test_search_result_with_duplicate():
    print("\n--- 3.1 Testing Search with Duplicate ---")
    url = f"{API_URL}/search"
    try:
        query_file = next(BASE_DIR.glob("*_duplicate.wav"))
        if not query_file.exists():
            print("No duplicate file found for testing.")
            return
    except StopIteration:
        return

    with open(query_file, 'rb') as f:
        files = {'file': (query_file.name, f, 'audio/wav')}
        response = requests.post(url, files=files, data={'top_k': 5})
    
    print(f"Search Status: {response.status_code}")
    print(f"Search results for duplicate file '{query_file.name}':\n")
    
    for item in response.json():
        print(f"- {item['filename']} (Score: {item['similarity_score']:.4f})")    
        
def test_multiple_audio_files_upload():
    """
    Requirement: Accepts multiple audio files and generates embeddings.
    This test verifies that the /embeddings endpoint handles a batch of files
    and returns a JSON object keyed by filename.
    """
    print("\n--- Testing Requirement: Multiple Audio File Upload ---")
    url = f"{API_URL}/embeddings"
    
    # 1. Identify all audio files in the integration_tests directory
    audio_extensions = {".wav", ".mp3"}
    
    # Check for suffixes in a case-insensitive manner
    audio_paths = [f for f in BASE_DIR.iterdir() if f.suffix.lower() in audio_extensions]
    
    if len(audio_paths) < 2:
        print("Skipping: This test requires at least 2 audio files in the folder.")
        return

    # 2. Prepare the batch of files
    files_to_upload = []
    opened_files = []
    
    print(f"Uploading {len(audio_paths)} files simultaneously...")
    for path in audio_paths:
        f = open(path, 'rb')
        opened_files.append(f)
        # The key 'files' must match the FastAPI parameter name: list[UploadFile] = File(...)
        files_to_upload.append(('files', (path.name, f, 'audio/wav')))

    try:
        # 3. Execute the multi-file POST request
        response = requests.post(url, files=files_to_upload)
        
        # 4. Validate the response structure
        if response.status_code == 200:
            data = response.json()
            print(f"Status Code: {response.status_code} (Success)")
            
            # Check if all files are represented in the response keys
            for path in audio_paths:
                if path.name in data:
                    print(f"  [✓] {path.name}: {data[path.name].get('status')}")
                else:
                    print(f"  [✗] {path.name}: Missing from response")
                    
            print(f"\nFull Response:\n{json.dumps(data, indent=2)}")
        else:
            print(f"Status Code: {response.status_code} (Failed)")
            print(f"Error: {response.text}")

    finally:
        # Always close files to prevent resource leaks
        for f in opened_files:
            f.close()        
        
def test_delete_sequence():
    print("\n--- 4. Testing Deletion ---")
    
    # 1. Get the list of IDs
    list_url = f"{API_URL}/embeddings"
    records = requests.get(list_url).json()
    
    if not records:
        print("Nothing to delete.")
        return

    # 2. Delete the first record by ID
    target_id = records[0]['id']
    target_name = records[0]['filename']
    print(f"Attempting to delete: {target_name} ({target_id[:8]}...)")
    
    delete_url = f"{API_URL}/embeddings/{target_id}"
    resp = requests.delete(delete_url)
    print(f"Delete ID Status: {resp.status_code} - {resp.json().get('message')}")

    # 3. Wipe all remaining records
    print("Attempting to wipe all records...")
    wipe_resp = requests.delete(list_url)
    print(f"Wipe All Status: {wipe_resp.status_code} - {wipe_resp.json().get('message')}")

if __name__ == "__main__":
    #test_upload_all_in_folder()
    test_multiple_audio_files_upload()
    test_invalid_file_rejection()
    test_list_all_files()
    test_search_with_first_found()
    test_search_result_with_duplicate()
    # Adding the delete test at the end to clean up
    test_delete_sequence()