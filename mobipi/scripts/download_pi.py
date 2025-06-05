#!/usr/bin/env python3
"""
Script to download and extract PI policy checkpoints for selected environments and seeds.

Usage:
    python download_pi.py [--nav]

Options:
    --nav    Download the BC w/ Nav checkpoints instead of the default non-mobile ones.

@yjy0625
"""
import os
import sys
import json
import zipfile
import argparse
from urllib.parse import urljoin

import requests
from tqdm import tqdm
from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR

# Default Base URL and file list for PI checkpoints
BASE_URL = "https://download.cs.stanford.edu/juno/mobipi/pi/"
FILES = [
    "04-12-CloseDrawer_seed_1_CloseDrawer_mg-300.zip",
    "04-12-CloseDrawer_seed_2_CloseDrawer_mg-300.zip",
    "04-12-CloseDrawer_seed_3_CloseDrawer_mg-300.zip",
    "04-12-CloseSingleDoor_seed_1_CloseSingleDoor_mg-300.zip",
    "04-12-CloseSingleDoor_seed_2_CloseSingleDoor_mg-300.zip",
    "04-12-CloseSingleDoor_seed_3_CloseSingleDoor_mg-300.zip",
    "04-12-TurnOnFaucet_seed_1_TurnOnSinkFaucet_mg-300.zip",
    "04-12-TurnOnFaucet_seed_2_TurnOnSinkFaucet_mg-300.zip",
    "04-12-TurnOnFaucet_seed_3_TurnOnSinkFaucet_mg-300.zip",
    "04-12-TurnOnMicrowave_seed_1_TurnOnMicrowave_mg-300.zip",
    "04-12-TurnOnMicrowave_seed_2_TurnOnMicrowave_mg-300.zip",
    "04-12-TurnOnMicrowave_seed_3_TurnOnMicrowave_mg-300.zip",
    "04-12-TurnOnStove_seed_1_TurnOnStove_mg-300.zip",
    "04-12-TurnOnStove_seed_2_TurnOnStove_mg-300.zip",
    "04-12-TurnOnStove_seed_3_TurnOnStove_mg-300.zip",
]

# Data zip mapping by environment name
DATA_ZIPS = {
    "CloseDrawer":     "data_CloseDrawer.zip",
    "CloseSingleDoor": "data_CloseSingleDoor.zip",
    "TurnOnFaucet":    "data_TurnOnSinkFaucet.zip",
    "TurnOnMicrowave": "data_TurnOnMicrowave.zip",
    "TurnOnStove":     "data_TurnOnStove.zip",
}

# Mapping for checking extracted dataset paths
DATA_DIR_MAP = {
    "CloseDrawer":     ("kitchen_drawer",    "CloseDrawer"),
    "CloseSingleDoor": ("kitchen_doors",     "CloseSingleDoor"),
    "TurnOnFaucet":    ("kitchen_sink",      "TurnOnSinkFaucet"),
    "TurnOnMicrowave": ("kitchen_microwave", "TurnOnMicrowave"),
    "TurnOnStove":     ("kitchen_stove",     "TurnOnStove"),
}

# Parse command-line args
parser = argparse.ArgumentParser(
    description="Download & extract PI policy checkpoints; pass --nav to grab BCNav versions"
)
parser.add_argument(
    "--nav",
    action="store_true",
    help="Download the ‘bcnav’ checkpoints (04-15-…-nav.zip) instead of the default PI ones"
)
args = parser.parse_args()

# Override if --nav is specified
if args.nav:
    BASE_URL = "https://download.cs.stanford.edu/juno/mobipi/bcnav/"
    FILES = [
        "04-15-CloseDrawer_seed_1_CloseDrawer_mg-1500-nav.zip",
        "04-15-CloseDrawer_seed_2_CloseDrawer_mg-1500-nav.zip",
        "04-15-CloseDrawer_seed_3_CloseDrawer_mg-1500-nav.zip",
        "04-15-CloseSingleDoor_seed_1_CloseSingleDoor_mg-1500-nav.zip",
        "04-15-CloseSingleDoor_seed_2_CloseSingleDoor_mg-1500-nav.zip",
        "04-15-CloseSingleDoor_seed_3_CloseSingleDoor_mg-1500-nav.zip",
        "04-15-TurnOnMicrowave_seed_1_TurnOnMicrowave_mg-1500-nav.zip",
        "04-15-TurnOnMicrowave_seed_2_TurnOnMicrowave_mg-1500-nav.zip",
        "04-15-TurnOnMicrowave_seed_3_TurnOnMicrowave_mg-1500-nav.zip",
        "04-15-TurnOnSinkFaucet_seed_1_TurnOnSinkFaucet_mg-1500-nav.zip",
        "04-15-TurnOnSinkFaucet_seed_2_TurnOnSinkFaucet_mg-1500-nav.zip",
        "04-15-TurnOnSinkFaucet_seed_3_TurnOnSinkFaucet_mg-1500-nav.zip",
        "04-15-TurnOnStove_seed_1_TurnOnStove_mg-1500-nav.zip",
        "04-15-TurnOnStove_seed_2_TurnOnStove_mg-1500-nav.zip",
        "04-15-TurnOnStove_seed_3_TurnOnStove_mg-1500-nav.zip",
    ]

# Extract unique env names and seed values for prompting
envs = sorted({fname.split('-', 2)[2].split('_seed')[0] for fname in FILES})
seeds = sorted({fname.split('_seed_')[1].split('_')[0] for fname in FILES}, key=int)


def prompt_selection(options, prompt_text):
    print(prompt_text)
    for idx, opt in enumerate(options, start=1):
        print(f"  {idx}) {opt}")
    while True:
        choice = input("Enter numbers separated by comma, or 'all': ").strip().lower()
        if choice in ('all', 'a', '*'):
            return options
        parts = [c.strip() for c in choice.split(',')]
        selected = []
        try:
            for p in parts:
                i = int(p)
                if 1 <= i <= len(options):
                    selected.append(options[i-1])
                else:
                    raise ValueError
            if selected:
                return selected
        except ValueError:
            pass
        print("Invalid input. Please try again.")


def download_file(url, local_path, desc):
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get('Content-Length', 0)) or None
    with open(local_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=desc
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_and_extract_zip(url, local_zip, extract_root):
    if os.path.exists(local_zip) and not zipfile.is_zipfile(local_zip):
        os.remove(local_zip)
    if not os.path.exists(local_zip):
        download_file(url, local_zip, os.path.basename(local_zip))
    try:
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(extract_root)
    except zipfile.BadZipFile:
        os.remove(local_zip)
        download_file(url, local_zip, os.path.basename(local_zip))
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(extract_root)
    os.remove(local_zip)


def ensure_datasets(envs_needed, nav=False):
    for env in envs_needed:
        data_zip = DATA_ZIPS.get(env)
        if not data_zip:
            continue
        local_zip = os.path.join(DATA_ROOT_DIR, data_zip)
        subdir, envdir = DATA_DIR_MAP[env]
        check_path = os.path.join(DATA_ROOT_DIR, 'v0.1', 'single_stage', subdir, envdir, 'mg_nav' if nav else 'mg')
        if os.path.isdir(check_path):
            print(f"Dataset for {env} already exists at {check_path}, skipping.")
            continue
        print(f"Downloading dataset for {env}...")
        url = urljoin(BASE_URL, data_zip)
        download_and_extract_zip(url, local_zip, DATA_ROOT_DIR)


def download_and_extract_checkpoint(fname, dest_root):
    url = urljoin(BASE_URL, fname)
    local_zip = os.path.join(dest_root, fname)
    folder1 = fname.split("/")[-1].split("_")[0]
    folder2 = "_".join(fname.split("/")[-1][:-4].split("_")[1:])
    extract_dir = os.path.join(dest_root, folder1, folder2)
    if os.path.isdir(extract_dir):
        print(f"Skipping {fname}: '{extract_dir}' already exists.")
        return
    download_and_extract_zip(url, local_zip, dest_root)
    fix_config_paths_for_zip(fname, dest_root)


def fix_config_paths_for_zip(zip_fname: str, dest_root: str):
    stem = os.path.splitext(zip_fname)[0]
    try:
        prefix, suffix = stem.split("_", 1)
    except ValueError:
        print(f"[WARN] Unexpected zip name format: {zip_fname}")
        return
    base = os.path.join(dest_root, prefix, suffix)
    print(f"Fixing {base}")

    for dirpath, _, filenames in os.walk(base):
        if "config.json" not in filenames:
            continue

        cfg_path = os.path.join(dirpath, "config.json")
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"[WARN] Failed to read JSON at {cfg_path}; skipping.")
            continue

        old_p = cfg.get("train", {}).get("data", [{}])[0].get("path", "")
        if "datasets" not in old_p:
            continue

        _, suffix = old_p.split("datasets", 1)
        new_p = os.path.join(DATA_ROOT_DIR, suffix.lstrip(os.sep))

        cfg["train"]["data"][0]["path"] = new_p
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[INFO] Patched config path in {cfg_path} → {new_p}")


if __name__ == '__main__':
    mode = "BC w/ Nav" if args.nav else "Source Policies"
    print(f"Available environments ({mode}):")
    selected_envs = prompt_selection(envs, "Select environment(s):")
    print("Available seeds:")
    selected_seeds = prompt_selection(seeds, "Select seed(s):")

    # Ensure root dirs exist
    os.makedirs(os.path.join(POLICY_CKPT_ROOT_DIR, 'robocasa', 'bc_xfmr_nav' if args.nav else 'bc_xfmr'), exist_ok=True)
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)

    initial = [
        f for f in FILES
        if f.split('-', 2)[2].split('_seed')[0] in selected_envs
           and f.split('_seed_')[1].split('_')[0] in selected_seeds
    ]

    to_download = []
    for fname in initial:
        extract_dir = os.path.join(
            POLICY_CKPT_ROOT_DIR, 'robocasa', 'bc_xfmr_nav' if args.nav else 'bc_xfmr', fname[:-4]
        )
        if os.path.isdir(extract_dir):
            print(f"Checkpoint {fname} already extracted, skipping.")
        else:
            to_download.append(fname)

    if not to_download:
        print("All selected checkpoints are already downloaded and extracted. Nothing to do.")
        sys.exit(0)

    if args.nav:
        print(f"Will download and extract {len(to_download)} checkpoint(s). Continue?")
    else:
        print(f"Will download and extract {len(to_download)} checkpoint(s) plus their datasets. Continue?")
    while True:
        ans = input("[y/n]: ").strip().lower()
        if ans in ('y', 'yes'):
            break
        if ans in ('n', 'no'):
            print("Aborted by user.")
            sys.exit(0)
        print("Please enter 'y' or 'n'.")

    envs_needed = {
        fname.split('-', 2)[2].split('_seed')[0] for fname in to_download
    }
    ensure_datasets(envs_needed, nav=args.nav)

    for fname in to_download:
        download_and_extract_checkpoint(
            fname,
            os.path.join(POLICY_CKPT_ROOT_DIR, 'robocasa', 'bc_xfmr_nav' if args.nav else 'bc_xfmr')
        )

    print("All downloads and extractions completed.")
