from pathlib import Path

ROOT = Path(__file__).resolve().parent     # the F1 App folder

CACHE_DIR = ROOT / "fastf1_cache"          # FastF1's downloads
DATA_DIR = ROOT / "data"                   # your own data files
RAW_DIR = DATA_DIR / "raw"                 # one file per race (piece 2)
OPENF1_DIR = DATA_DIR / "openf1"           # one file per race from OpenF1 (2023 onwards)
OPENF1_JSON_DIR = OPENF1_DIR / "json"      # OpenF1's answers, saved so we never download twice
MODELS_DIR = ROOT / "models"               # trained models
FIGURES_DIR = ROOT / "figures"             # charts for the dissertation

SEASONS = [2022, 2023, 2024, 2025, 2026]   # which years to download