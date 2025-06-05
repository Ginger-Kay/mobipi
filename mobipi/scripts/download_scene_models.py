"""
Script to download pre-trained 3DGS models.

Usage:
    python download_scene_models.py

@yjy0625
"""
import os
import sys
import zipfile
from urllib.parse import urljoin

import requests
from tqdm import tqdm

import sys, os
print(" TODO: sys.executable =", sys.executable)
print(" TODO: sys.path =", "\n    ".join(sys.path))

from mobipi.macros import SCENE_MODEL_ROOT_DIR
from mobipi.utils.constants import TRAIN_LAYOUTS, TEST_LAYOUTS

# Base URL for scene model ZIP files
BASE_URL = "https://download.cs.stanford.edu/juno/mobipi/gs/"
# Environment (task) names
ENVS = [
    "close_drawer",
    "close_single_door",
    "turn_on_sink_faucet",
    "turn_on_microwave",
    "turn_on_stove",
]
# Scene indices (layout/style IDs)
SCENES = list(range(10))  # 0 through 9


def prompt_selection(options, prompt_text, is_scene=False):
    """
    Prompt the user to select one or more options by index or 'all'.
    Returns a list of selected option indices.
    """
    print(prompt_text)
    for idx, name in enumerate(options):
        print(f"  [{idx}] {name}")
    while True:
        if is_scene:
            choice = input("Enter indices (comma-separated) or 'all' or 'train' (all train scenes) or 'test' (all test scenes): ").strip().lower()
        else:
            choice = input("Enter indices (comma-separated) or 'all': ").strip().lower()
        if choice == 'all':
            return list(range(len(options)))
        elif is_scene and choice == 'train':
            return TRAIN_LAYOUTS
        elif is_scene and choice == 'test':
            return TEST_LAYOUTS
        parts = [p.strip() for p in choice.split(',') if p.strip()]
        try:
            indices = sorted({int(p) for p in parts})
        except ValueError:
            print("Invalid input: please enter numbers or 'all'.")
            continue
        if any(i < 0 or i >= len(options) for i in indices):
            print("One or more indices out of range. Try again.")
            continue
        return indices


def prompt_yes_no(prompt_text):
    """
    Prompt the user for a yes/no answer. Returns True for yes, False for no.
    """
    while True:
        ans = input(f"{prompt_text} [y/n]: ").strip().lower()
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False
        print("Please answer with 'y' or 'n'.")


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def download_and_extract(filename):
    """
    Download a ZIP file if needed, extract it into the root directory,
    then optionally remove the ZIP.
    Skips download if already extracted.
    """
    # Define paths
    dest_zip = os.path.join(SCENE_MODEL_ROOT_DIR, filename)
    folder1 = "_".join(filename.split("/")[-1].split("_")[:2])
    folder2 = "_".join(filename.split("/")[-1][:-4].split("_")[2:])
    extract_marker = os.path.join(SCENE_MODEL_ROOT_DIR, folder1, folder2)

    # If extracted output already exists, skip
    if os.path.isdir(extract_marker):
        print(f"Skipping {filename}, content already extracted.")
        return

    # If zip exists, check integrity
    if os.path.isfile(dest_zip):
        if zipfile.is_zipfile(dest_zip):
            print(f"Using existing ZIP for {filename}.")
        else:
            print(f"Existing {filename} is corrupted. Re-downloading.")
            os.remove(dest_zip)
    # Download if ZIP missing
    if not os.path.isfile(dest_zip):
        url = urljoin(BASE_URL, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(dest_zip, 'wb') as f, tqdm(
            desc=f"Downloading {filename}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    # Extract
    try:
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(SCENE_MODEL_ROOT_DIR)
    except zipfile.BadZipFile:
        print(f"Failed to extract {filename}: invalid zip. Removing and retrying.")
        os.remove(dest_zip)
        download_and_extract(filename)
        return

    # Remove ZIP file
    os.remove(dest_zip)


def main():
    # Ensure root directory exists
    ensure_dir(SCENE_MODEL_ROOT_DIR)

    # Prompt for environment selection
    env_indices = prompt_selection(
        ENVS,
        "Select environment(s) to download scene models for:"
    )
    selected_envs = [ENVS[i] for i in env_indices]

    # Prompt for scene selection
    scene_indices = prompt_selection(
        [f"layout{idx}_style{idx}" for idx in SCENES],
        "Select scene(s) (layout/style) to download:",
        is_scene=True
    )
    selected_scenes = [SCENES[i] for i in scene_indices]

    # Build filenames
    files = []
    for env in selected_envs:
        for sc in selected_scenes:
            files.append(f"{env}_layout{sc}_style{sc}.zip")

    total_size_gb = len(files)
    print(f"Will download a total of {total_size_gb} GB of data ({len(files)} files).")
    if not prompt_yes_no("Continue?"):
        print("Aborted by user.")
        sys.exit(0)

    # Download and extract each file
    for fname in files:
        try:
            download_and_extract(fname)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            break

    print("All downloads and extractions complete.")


if __name__ == "__main__":
    main()
