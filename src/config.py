from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent     # project folder (the code is in src/)

CACHE_DIR = ROOT / "fastf1_cache"          # FastF1 downloads
DATA_DIR = ROOT / "data"                   # our data files
RAW_DIR = DATA_DIR / "raw"                 # one file per race
OPENF1_DIR = DATA_DIR / "openf1"           # one file per race from OpenF1 (2023 on)
OPENF1_JSON_DIR = OPENF1_DIR / "json"      # OpenF1 replies, saved so we never download twice
MODELS_DIR = ROOT / "models"               # trained models
FIGURES_DIR = ROOT / "figures"             # charts for the dissertation

SEASONS = [2022, 2023, 2024, 2025, 2026]   # which years to download