"""Download the Judionline dataset from Kaggle into data/raw/.

Usage:
    python -m src.download
"""
import io
import urllib.request
import zipfile

from src.config import DATA_DIR, DATA_PATH, DATASET_URL


def download_dataset(force=False):
    """Download and extract the dataset, renaming the CSV to judionline.csv"""
    if DATA_PATH.exists() and not force:
        print(f"Dataset already exists: {DATA_PATH}")
        return DATA_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
    DATA_PATH.write_bytes(archive.read(csv_name))
    print(f"Saved to {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    download_dataset()
