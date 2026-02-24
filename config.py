"""Central configuration: paths, API endpoints, model defaults."""

from __future__ import annotations
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = ROOT / "reports"

# Ensure dirs exist on import
for _d in [RAW_DIR, PAPERS_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── LLM defaults ─────────────────────────────────────────────────────────────
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# ── Materials Project ─────────────────────────────────────────────────────────
MP_API_KEY = os.getenv("MP_API_KEY", "")
MP_NUM_ELEMENTS_MIN = 4
MP_NUM_ELEMENTS_MAX = 6
MP_MAX_RESULTS = 500

# ── AFLOW ─────────────────────────────────────────────────────────────────────
AFLOW_MAX_RESULTS = 200

# ── arXiv ─────────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES = ["cond-mat.mtrl-sci", "physics.comp-ph"]
ARXIV_MAX_RESULTS = 20

# ── Featurization ─────────────────────────────────────────────────────────────
FEATURIZER_SETS = {
    "minimal": ["Meredig"],              # ~20 features, fast
    "standard": ["ElementProperty", "YangSolidSolution", "ValenceOrbital"],  # ~80
    "comprehensive": ["ElementProperty", "YangSolidSolution", "ValenceOrbital",
                      "Miedema", "AtomicOrbitals"],  # ~200
}

# ── ML models available for AutoML ───────────────────────────────────────────
AUTOML_MODELS = [
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "GradientBoosting",
    "SVR",
    "Ridge",
]

AUTOML_DEFAULT_TRIALS = 50
AUTOML_DEFAULT_CV_FOLDS = 5
AUTOML_TIMEOUT_SEC = 600

# ── HEA target properties ─────────────────────────────────────────────────────
HEA_PROPERTIES = [
    "hardness",
    "yield_strength",
    "tensile_strength",
    "elongation",
    "thermal_stability",
    "corrosion_resistance",
    "formation_energy",
    "phase",
]
