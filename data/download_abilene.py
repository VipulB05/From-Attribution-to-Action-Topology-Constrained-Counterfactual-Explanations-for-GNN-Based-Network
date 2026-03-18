# download_abilene.py

"""
Download real Abilene dataset.
Run this once before training.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path


def download_abilene_from_kaggle():
    """
    Download Abilene dataset from Kaggle.
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials: https://www.kaggle.com/docs/api
    """
    print("Downloading Abilene dataset from Kaggle...")
    
    os.makedirs('data/raw', exist_ok=True)
    
    # Download using Kaggle API
    import kaggle
    kaggle.api.dataset_download_files(
        'dedyvanhauten/abilene',
        path='data/raw',
        unzip=True
    )
    
    print("Download complete!")


def download_abilene_manual():
    """
    Manual download instructions if Kaggle API doesn't work.
    """
    print("""
    Manual Download Instructions:
    
    1. Go to: https://www.kaggle.com/datasets/dedyvanhauten/abilene
    2. Click "Download" (requires Kaggle account)
    3. Extract the ZIP file
    4. Place files in: data/raw/
    
    OR
    
    1. Go to: https://ieee-dataport.org/documents/traffic-datsets-abilene-geant-taxibj
    2. Download the Abilene dataset
    3. Place in: data/raw/
    """)


if __name__ == "__main__":
    try:
        download_abilene_from_kaggle()
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
        print("\nFalling back to manual instructions...")
        download_abilene_manual()