"""Matminer-based HEA featurization tools."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from config import PROCESSED_DIR, FEATURIZER_SETS


FEATURE_TOOL_DEFINITIONS = [
    {
        "name": "featurize_compositions",
        "description": (
            "Convert HEA composition strings (e.g., 'Al0.2CoCrFeNi') into numerical "
            "feature vectors using matminer. Generates element statistics, mixing parameters, "
            "and HEA-specific descriptors (Yang omega, delta, VEC, Miedema). "
            "Input: cleaned .parquet file with 'composition' column. "
            "Output: featurized .parquet file ready for ML."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to cleaned .parquet file with 'composition' column",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the target property column",
                },
                "featurizer_set": {
                    "type": "string",
                    "description": (
                        "'minimal' (~20 features, fast), "
                        "'standard' (~80 features, recommended), "
                        "'comprehensive' (~200 features, slow)"
                    ),
                },
            },
            "required": ["input_path", "target_column"],
        },
    },
]


def featurize_compositions(
    input_path: str,
    target_column: str,
    featurizer_set: str = "standard",
) -> str:
    """Featurize HEA compositions using matminer."""
    try:
        import pandas as pd
        import numpy as np
        from pymatgen.core import Composition
        from matminer.featurizers.composition import (
            ElementProperty,
            YangSolidSolution,
            ValenceOrbital,
            Miedema,
            AtomicOrbitals,
        )
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.conversions import StrToComposition
    except ImportError as e:
        return f"ERROR: Missing package: {e}. Run: pip install matminer pymatgen"

    path = Path(input_path)
    if not path.exists():
        return f"ERROR: File not found: {input_path}"

    # Load
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        return f"ERROR: Unsupported format {path.suffix}"

    if "composition" not in df.columns:
        return f"ERROR: 'composition' column not found. Got: {list(df.columns)}"

    if target_column not in df.columns:
        return f"ERROR: Target column '{target_column}' not found."

    # Convert composition strings to pymatgen Composition objects
    stc = StrToComposition(target_col_id="composition_obj")
    try:
        df = stc.featurize_dataframe(df, "composition", ignore_errors=True)
    except Exception as e:
        return f"ERROR converting compositions: {e}"

    # Drop rows where conversion failed
    df = df.dropna(subset=["composition_obj"])
    if len(df) == 0:
        return "ERROR: All compositions failed to parse. Check format (e.g., 'Al0.5CoCrFeNi')"

    # Select featurizers
    fset = FEATURIZER_SETS.get(featurizer_set, FEATURIZER_SETS["standard"])
    featurizers = []

    if "ElementProperty" in fset or "Meredig" in fset:
        ep = ElementProperty.from_preset("magpie")
        ep.set_n_jobs(1)
        featurizers.append(ep)

    if "YangSolidSolution" in fset:
        yss = YangSolidSolution()
        yss.set_n_jobs(1)
        featurizers.append(yss)

    if "ValenceOrbital" in fset:
        vo = ValenceOrbital()
        vo.set_n_jobs(1)
        featurizers.append(vo)

    if "Miedema" in fset:
        try:
            m = Miedema()
            m.set_n_jobs(1)
            featurizers.append(m)
        except Exception:
            pass  # Miedema sometimes fails; skip gracefully

    if "AtomicOrbitals" in fset:
        try:
            ao = AtomicOrbitals()
            ao.set_n_jobs(1)
            featurizers.append(ao)
        except Exception:
            pass

    if not featurizers:
        return f"ERROR: No featurizers selected for set '{featurizer_set}'"

    mf = MultipleFeaturizer(featurizers)
    try:
        df = mf.featurize_dataframe(df, "composition_obj", ignore_errors=True)
    except Exception as e:
        return f"ERROR during featurization: {e}"

    # Drop helper column
    df = df.drop(columns=["composition_obj"], errors="ignore")

    # Drop non-numeric feature columns (keep composition + target)
    feature_cols = [
        c for c in df.columns
        if c not in ("composition", target_column) and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    keep_cols = ["composition", target_column] + feature_cols
    df = df[[c for c in keep_cols if c in df.columns]]

    # Drop rows with NaN features
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DIR / f"features_{target_column}_{ts}.parquet"
    df.to_parquet(out_path, index=False)

    return json.dumps({
        "n_samples": len(df),
        "n_samples_dropped": n_before - len(df),
        "n_features": len(feature_cols),
        "feature_names": feature_cols[:20],  # preview first 20
        "total_feature_names": feature_cols,
        "target_column": target_column,
        "featurized_path": str(out_path),
        "featurizer_set": featurizer_set,
    }, default=str)


FEATURE_TOOL_CALLABLES = {
    "featurize_compositions": featurize_compositions,
}
