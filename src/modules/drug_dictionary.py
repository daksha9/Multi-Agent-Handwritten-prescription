import os
from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "indian_medicines.csv"


def load_drug_table(csv_path: str = str(DEFAULT_CSV_PATH)) -> pd.DataFrame:
    """Returns a DataFrame with at least a 'name' column, deduplicated and cleaned."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Drug dictionary not found at {csv_path}.\n"
            "Re-download DATA/indian_medicine_data.csv from "
            "https://github.com/junioralive/Indian-Medicine-Dataset, "
            "or point csv_path at your own medicine list with at least a 'name' column."
        )

    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError("Drug dictionary CSV must have a 'name' column.")

    df = df.dropna(subset=["name"]).drop_duplicates(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()
    return df.reset_index(drop=True)
