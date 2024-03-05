import json
import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_datasets(json_file_path):
    """
    Reads a JSON file containing Hugging Face dataset IDs and descriptions,
    checks if each dataset is already downloaded, and if not, downloads it.

    Args:
        json_file_path (str): The path to the JSON file containing dataset IDs and descriptions.
    """
    # Load dataset information from the provided JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f).get('datasets', [])

    for dataset in datasets:
        dataset_id = dataset.get('id')
        # Assume the dataset is stored in a file named after the dataset ID
        local_file_path = f"./{dataset_id}.zip"

        if os.path.exists(local_file_path):
            print(f"Dataset '{dataset_id}' is already downloaded. Skipping...")
            continue

        print(f"Downloading dataset '{dataset_id}'...")
        # Download the dataset using hf_hub_download
        # The actual filename on the hub needs to be known or assumed, here it is assumed to be dataset_id.zip
        # hf_hub_download(repo_id=dataset_id, cache_dir='.', repo_type='dataset')
        snapshot_download(repo_id=dataset_id,  cache_dir='.', repo_type='dataset')

    print("All datasets processed.")

# Assuming 'datasets.json' is in the current directory
json_file_path = 'datasets.json'
download_datasets(json_file_path)
