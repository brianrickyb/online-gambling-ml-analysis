"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "judionline.csv"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/yaemico/judionline"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "judol_classifier.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
PREDICTION_LOG_PATH = ROOT / "logs" / "predictions.jsonl"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TEXT_COL = "komentar"
LABEL_COL = "label"
# Output of someone else's model; using it as a feature would be data leakage.
LEAKY_COLS = ["predicted_label", "komentar_clean"]

LABEL_NAMES = {0: "normal", 1: "judol"}
# Score >= THRESHOLD is classified as judol. Raise it to delete fewer normal
# comments, lower it to let less spam through (see notebooks/02_modeling.ipynb).
THRESHOLD = 0.5
RANDOM_STATE = 42
TEST_SIZE = 0.2
