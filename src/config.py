"""
config.py — Central configuration: all paths, constants, and model parameters.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
IMAGES_DIR     = ROOT / "images"          # committed to GitHub for README

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
RAW_DATA_FILE        = DATA_RAW / "CC GENERAL.csv"
CLEAN_DATA_FILE      = DATA_PROCESSED / "cc_clean_scaled.csv"
UNSCALED_DATA_FILE   = DATA_PROCESSED / "cc_clean_unscaled.csv"

# ── Feature Groups ─────────────────────────────────────────────────────────────
ID_COLUMN = "CUST_ID"

MONETARY_FEATURES = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
]
FREQUENCY_FEATURES = [
    "BALANCE_FREQUENCY", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT",
]
COUNT_FEATURES     = ["CASH_ADVANCE_TRX", "PURCHASES_TRX", "TENURE"]

ENGINEERED_FEATURES = [
    "PURCHASES_TO_LIMIT_RATIO", "CASH_ADVANCE_RATIO",
    "PAYMENT_TO_MINIMUM_RATIO", "MONTHLY_AVG_PURCHASE",
    "INSTALLMENT_TO_PURCHASE_RATIO", "CASH_ADVANCE_TO_CREDIT_RATIO",
    "BALANCE_TO_CREDIT_RATIO",
]

# ── Clustering ─────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
N_CLUSTERS     = 4
K_RANGE        = range(2, 11)
PCA_COMPONENTS = 2

# ── Plot style ─────────────────────────────────────────────────────────────────
PALETTE    = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
FIG_DPI    = 150
