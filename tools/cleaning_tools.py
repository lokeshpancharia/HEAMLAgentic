"""Dataset cleaning tools: deduplication, normalization, outlier removal."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import PROCESSED_DIR, RAW_DIR


CLEANING_TOOL_DEFINITIONS = [
    {
        "name": "merge_raw_data",
        "description": (
            "Merge all JSON files from data/raw/ into a single unified dataset. "
            "Handles files from Materials Project, AFLOW, and PDF extraction. "
            "Returns the path to the merged dataset and a count of records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_dir": {
                    "type": "string",
                    "description": "Path to raw data directory (default: data/raw)",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the target property column (e.g., 'hardness')",
                },
            },
            "required": ["target_column"],
        },
    },
    {
        "name": "clean_dataset",
        "description": (
            "Clean a dataset CSV/JSON file: remove duplicates, handle missing values, "
            "normalize units, and remove outliers using IQR method. "
            "Returns path to cleaned .parquet file and summary statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to input CSV or JSON dataset file",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the target property column",
                },
                "composition_column": {
                    "type": "string",
                    "description": "Name of composition column (default: 'composition')",
                },
                "iqr_factor": {
                    "type": "number",
                    "description": "IQR multiplier for outlier detection (default: 3.0)",
                },
                "impute_strategy": {
                    "type": "string",
                    "description": "Strategy for missing values: 'median', 'mean', 'drop' (default: 'median')",
                },
            },
            "required": ["input_path", "target_column"],
        },
    },
]


def merge_raw_data(
    target_column: str,
    raw_dir: str | None = None,
) -> str:
    """Merge all raw JSON files into a unified dataset."""
    try:
        import pandas as pd
    except ImportError:
        return "ERROR: pandas not installed"

    raw_path = Path(raw_dir) if raw_dir else RAW_DIR
    json_files = list(raw_path.glob("*.json"))

    if not json_files:
        return json.dumps({"error": "No JSON files found in raw directory", "n_records": 0})

    all_records = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, list):
                all_records.extend(data)
            elif isinstance(data, dict):
                all_records.append(data)
        except Exception:
            continue

    if not all_records:
        return json.dumps({"error": "No records loaded from raw files", "n_records": 0})

    df = pd.DataFrame(all_records)

    # Standardize composition column
    comp_cols = [c for c in df.columns if "composition" in c.lower() or "formula" in c.lower()]
    if comp_cols and "composition" not in df.columns:
        df = df.rename(columns={comp_cols[0]: "composition"})

    # Save merged
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"combined_raw_{ts}.csv"
    df.to_csv(out_path, index=False)

    available_cols = [c for c in df.columns if c != "composition"]
    return json.dumps({
        "n_records": len(df),
        "columns": list(df.columns),
        "available_properties": available_cols,
        "merged_path": str(out_path),
        "note": f"Ensure target column '{target_column}' is present. Available: {available_cols}",
    }, default=str)


def clean_dataset(
    input_path: str,
    target_column: str,
    composition_column: str = "composition",
    iqr_factor: float = 3.0,
    impute_strategy: str = "median",
) -> str:
    """Clean dataset: dedup, unit normalize, impute, outlier removal."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return "ERROR: pandas/numpy not installed"

    path = Path(input_path)
    if not path.exists():
        return f"ERROR: File not found: {input_path}"

    # Load
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".json",):
        df = pd.read_json(path)
    elif path.suffix in (".parquet",):
        df = pd.read_parquet(path)
    elif path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        return f"ERROR: Unsupported file format: {path.suffix}"

    # Rename composition column
    if composition_column != "composition" and composition_column in df.columns:
        df = df.rename(columns={composition_column: "composition"})

    if "composition" not in df.columns:
        return f"ERROR: No 'composition' column found. Columns: {list(df.columns)}"

    if target_column not in df.columns:
        return (
            f"ERROR: Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    n_start = len(df)
    log: list[str] = []

    # Drop rows with missing composition or target
    df = df.dropna(subset=["composition", target_column])
    log.append(f"Dropped {n_start - len(df)} rows with missing composition/target")

    # Remove duplicate compositions (keep first)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["composition"])
    log.append(f"Removed {n_before_dedup - len(df)} duplicate compositions")

    # Convert target to numeric
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    # Impute other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if impute_strategy in ("median", "mean"):
        for col in numeric_cols:
            if col == target_column:
                continue
            fill_val = df[col].median() if impute_strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)

    # Outlier removal on target column using IQR
    q1 = df[target_column].quantile(0.25)
    q3 = df[target_column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    n_before_outlier = len(df)
    df = df[(df[target_column] >= lower) & (df[target_column] <= upper)]
    log.append(f"Removed {n_before_outlier - len(df)} outliers (IQR×{iqr_factor})")

    # Keep only composition + target + numeric columns
    keep_cols = ["composition", target_column] + [
        c for c in numeric_cols if c not in ("composition", target_column)
    ]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DIR / f"cleaned_{target_column}_{ts}.parquet"
    df.to_parquet(out_path, index=False)

    return json.dumps({
        "n_samples_before": n_start,
        "n_samples_after": len(df),
        "target_column": target_column,
        "target_stats": {
            "mean": float(df[target_column].mean()),
            "std": float(df[target_column].std()),
            "min": float(df[target_column].min()),
            "max": float(df[target_column].max()),
        },
        "columns": list(df.columns),
        "cleaned_path": str(out_path),
        "processing_log": log,
    }, default=str)


CLEANING_TOOL_CALLABLES = {
    "merge_raw_data": merge_raw_data,
    "clean_dataset": clean_dataset,
}
