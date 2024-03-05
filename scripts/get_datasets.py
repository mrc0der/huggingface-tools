"""get_datasets.py"""
import json
import os

from huggingface_hub import snapshot_download

DATASET_HOME = os.getenv('MODEL_HOME')
JSON_LIST = os.getenv('DATASET_FILE_LIST', 'datasets.json')


def download_datasets():
    """
    Reads a JSON file containing Hugging Face dataset IDs and descriptions,
    checks if each dataset is already downloaded, and if not, downloads it.
    """
    # Load dataset information from the provided JSON file
    with open(JSON_LIST, 'r', encoding='utf-8') as f:
        datasets = json.load(f).get('datasets', [])

    for dataset in datasets:
        dataset_id = dataset.get('id')
        # Assume the dataset is stored in a file named after the dataset ID
        local_file_path = f"{DATASET_HOME}/{dataset_id}.zip"

        if os.path.exists(local_file_path):
            print(f"Dataset '{dataset_id}' is already downloaded. Skipping...")
            continue

        print(f"Downloading dataset '{dataset_id}'...")
        # Download the dataset using hf_hub_download
        # The actual filename on the hub needs to be known or assumed, here it is assumed to be dataset_id.zip
        # hf_hub_download(repo_id=dataset_id, cache_dir='.', repo_type='dataset')
        snapshot_download(repo_id=dataset_id,  cache_dir=f'{DATASET_HOME}/{dataset_id}', repo_type='dataset')

    print("All datasets processed.")

download_datasets()
