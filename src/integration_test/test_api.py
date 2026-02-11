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
        query_file = next(BASE_DIR.glob("*.wav"))
    except StopIteration:
        return


    with open(query_file, 'rb') as f:
        files = {'file': (query_file.name, f, 'audio/wav')}
        response = requests.post(url, files=files, data={'top_k': 3})
    
    print(f"Search Status: {response.status_code}")
    
    for item in response.json():
        print(f"- {item['filename']} (Score: {item['similarity_score']:.4f})")
        
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
    test_upload_all_in_folder()
    test_list_all_files()
    test_search_with_first_found()
    # Adding the delete test at the end to clean up
    test_delete_sequence()