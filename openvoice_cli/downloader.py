from pathlib import Path 
from tqdm import tqdm
import requests

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

def create_directory_if_not_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True)

def download_checkpoint(dest_dir):
    # Define paths
    model_path = Path(dest_dir)

    # Define files and their corresponding URLs
    files_to_download = {
         "checkpoint.pth": f"https://huggingface.co/daswer123/openvoice-tunner-v2/resolve/main/checkpoint.pth?download=true",
         "config.json": f"https://huggingface.co/daswer123/openvoice-tunner-v2/raw/main/config.json",
    }

    # Check and create directories
    create_directory_if_not_exists(model_path)

    # Download files if they don't exist
    for filename, url in files_to_download.items():
         destination = model_path / filename
         if not destination.exists():
             print(f"[OpenVoice Converter] Downloading {filename}...")
             download_file(url, destination)