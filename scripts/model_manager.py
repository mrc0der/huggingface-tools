import json
import os
import sys
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

MODEL_HOME = os.getenv('MODEL_HOME')
MODEL_FILE_LIST = os.getenv('MODEL_FILE_LIST', 'models.json')

if MODEL_HOME is None:
    print('Please set MODEL_HOME and re-run')
    sys.exit(1)

def download_models(json_file_path):
    """
    Reads a JSON file containing Hugging Face model IDs, descriptions, and files,
    checks if each model file is already downloaded, and if not, downloads it.
    """
    # Load model information from the provided JSON file
    with open(MODEL_FILE_LIST, 'r') as f:
        models = json.load(f).get('models', [])

    for model in models:
        model_id = model.get('id')
        files = model.get('files', [])

        for file in files:
            local_file_path = f"{MODEL_HOME}/{model_id}/{file}"

            if os.path.exists(local_file_path):
                print(f"File '{file}' for model '{model_id}' is already downloaded. Skipping...")
                continue

            if not os.path.exists(f"{MODEL_HOME}/{model_id}"):
                os.makedirs(f"{MODEL_HOME}/{model_id}")

            print(f"Downloading file '{file}' for model '{model_id}'...")
            hf_hub_download(repo_id=model_id, filename=file, cache_dir=f"{MODEL_HOME}/{model_id}")

    print("All models and their files processed.")

download_models()
