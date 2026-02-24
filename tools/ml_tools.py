"""ML training, HPO, and evaluation tools."""

from __future__ import annotations
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from config import MODELS_DIR, AUTOML_DEFAULT_TRIALS, AUTOML_DEFAULT_CV_FOLDS, AUTOML_TIMEOUT_SEC

warnings.filterwarnings("ignore")


ML_TOOL_DEFINITIONS = [
    {
        "name": "run_model_comparison",
        "description": (
            "Run a quick 5-fold CV comparison of multiple ML models (RF, XGBoost, LightGBM, "
            "GradientBoosting, SVR, Ridge) with default hyperparameters. "
            "Returns a ranked leaderboard by R² score. Use this first to identify top candidates "
            "before running full hyperparameter optimization."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to featurized .parquet file",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the target property column",
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Models to compare: ['RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting', 'SVR', 'Ridge']",
                },
                "cv_folds": {
                    "type": "integer",
                    "description": "Number of CV folds (default 5)",
                },
                "task_type": {
                    "type": "string",
                    "description": "'regression' or 'classification' (default: 'regression')",
                },
            },
            "required": ["data_path", "target_column"],
        },
    },
    {
        "name": "optimize_hyperparameters",
        "description": (
            "Run Optuna hyperparameter optimization on a specific model. "
            "Tries n_trials parameter combinations, evaluates with CV, and returns best params. "
            "Use this on the top 2-3 models from run_model_comparison."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to featurized .parquet file",
                },
                "target_column": {"type": "string"},
                "model_type": {
                    "type": "string",
                    "description": "One of: 'RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting', 'SVR', 'Ridge'",
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Number of Optuna trials (default 50)",
                },
                "cv_folds": {"type": "integer"},
                "timeout_sec": {
                    "type": "integer",
                    "description": "Max seconds for optimization (default 600)",
                },
            },
            "required": ["data_path", "target_column", "model_type"],
        },
    },
    {
        "name": "train_final_model",
        "description": (
            "Train the final model on the full dataset using the best hyperparameters. "
            "Saves the trained model as a .joblib file with metadata. "
            "Returns the model path and final CV metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "target_column": {"type": "string"},
                "model_type": {"type": "string"},
                "best_params": {
                    "type": "object",
                    "description": "Hyperparameters from optimize_hyperparameters",
                },
                "cv_folds": {"type": "integer"},
            },
            "required": ["data_path", "target_column", "model_type", "best_params"],
        },
    },
]


def _load_data(data_path: str, target_column: str):
    """Load features and target from parquet file."""
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in {list(df.columns)}")

    feature_cols = [c for c in df.columns if c not in ("composition", target_column)]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_column].values.astype(np.float64)
    return X, y, feature_cols


def _get_model(model_type: str, params: dict | None = None):
    """Instantiate an sklearn-compatible model."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    params = params or {}

    if model_type == "RandomForest":
        return RandomForestRegressor(**{k: v for k, v in params.items()
                                        if k in RandomForestRegressor().get_params()},
                                     random_state=42, n_jobs=-1)
    elif model_type == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**{k: v for k, v in params.items()
                               if k in XGBRegressor().get_params()},
                            random_state=42, verbosity=0, n_jobs=-1)
    elif model_type == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**{k: v for k, v in params.items()
                                if k in LGBMRegressor().get_params()},
                             random_state=42, verbose=-1, n_jobs=-1)
    elif model_type == "GradientBoosting":
        return GradientBoostingRegressor(**{k: v for k, v in params.items()
                                            if k in GradientBoostingRegressor().get_params()},
                                         random_state=42)
    elif model_type == "SVR":
        return Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(**{k: v for k, v in params.items()
                                        if k in SVR().get_params()}))])
    elif model_type == "Ridge":
        return Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(**{k: v for k, v in params.items()
                                            if k in Ridge().get_params()}))])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_model_comparison(
    data_path: str,
    target_column: str,
    models: list[str] | None = None,
    cv_folds: int = AUTOML_DEFAULT_CV_FOLDS,
    task_type: str = "regression",
) -> str:
    """Quick CV comparison of multiple models."""
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import time

    if models is None:
        models = ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "SVR", "Ridge"]

    try:
        X, y, feature_cols = _load_data(data_path, target_column)
    except Exception as e:
        return f"ERROR loading data: {e}"

    leaderboard = []
    for model_name in models:
        try:
            model = _get_model(model_name)
            t0 = time.time()
            scores = cross_val_score(model, X, y, cv=cv_folds,
                                     scoring="r2", n_jobs=-1)
            rmse_scores = cross_val_score(model, X, y, cv=cv_folds,
                                          scoring="neg_root_mean_squared_error", n_jobs=-1)
            elapsed = time.time() - t0
            leaderboard.append({
                "model": model_name,
                "cv_r2_mean": float(np.mean(scores)),
                "cv_r2_std": float(np.std(scores)),
                "cv_rmse_mean": float(-np.mean(rmse_scores)),
                "time_sec": round(elapsed, 1),
            })
        except Exception as e:
            leaderboard.append({"model": model_name, "error": str(e)})

    # Sort by R²
    leaderboard = sorted(leaderboard, key=lambda x: x.get("cv_r2_mean", -999), reverse=True)

    return json.dumps({
        "leaderboard": leaderboard,
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "cv_folds": cv_folds,
        "best_model": leaderboard[0]["model"] if leaderboard else None,
        "recommendation": f"Run optimize_hyperparameters on top 2: {[m['model'] for m in leaderboard[:2]]}",
    }, default=str)


def optimize_hyperparameters(
    data_path: str,
    target_column: str,
    model_type: str,
    n_trials: int = AUTOML_DEFAULT_TRIALS,
    cv_folds: int = AUTOML_DEFAULT_CV_FOLDS,
    timeout_sec: int = AUTOML_TIMEOUT_SEC,
) -> str:
    """Optuna HPO for a given model type."""
    try:
        import optuna
        import numpy as np
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return "ERROR: optuna not installed. Run: pip install optuna"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        X, y, _ = _load_data(data_path, target_column)
    except Exception as e:
        return f"ERROR loading data: {e}"

    def objective(trial: optuna.Trial) -> float:
        if model_type == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            }
        elif model_type == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            }
        elif model_type == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        elif model_type == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        elif model_type == "SVR":
            params = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            }
        elif model_type == "Ridge":
            params = {"alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True)}
        else:
            raise ValueError(f"Unknown model: {model_type}")

        model = _get_model(model_type, params)
        scores = cross_val_score(model, X, y, cv=cv_folds,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        return float(-np.mean(scores))  # minimize RMSE

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    best_params = study.best_params
    best_rmse = study.best_value

    # Final R² with best params
    from sklearn.model_selection import cross_val_score
    best_model = _get_model(model_type, best_params)
    r2_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring="r2", n_jobs=-1)

    return json.dumps({
        "model_type": model_type,
        "best_params": best_params,
        "best_cv_rmse": float(best_rmse),
        "best_cv_r2": float(r2_scores.mean()),
        "n_trials_completed": len(study.trials),
        "recommendation": f"Run train_final_model with these params",
    }, default=str)


def train_final_model(
    data_path: str,
    target_column: str,
    model_type: str,
    best_params: dict[str, Any],
    cv_folds: int = AUTOML_DEFAULT_CV_FOLDS,
) -> str:
    """Train final model on full data and save to disk."""
    import joblib
    import numpy as np
    from sklearn.model_selection import cross_val_score

    try:
        X, y, feature_cols = _load_data(data_path, target_column)
    except Exception as e:
        return f"ERROR loading data: {e}"

    model = _get_model(model_type, best_params)

    # Final CV metrics
    r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2", n_jobs=-1)
    rmse_scores = cross_val_score(model, X, y, cv=cv_folds,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)

    # Train on full dataset
    model.fit(X, y)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"best_model_{target_column}_{ts}.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "target_column": target_column,
        "feature_names": feature_cols,
        "best_params": best_params,
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
        "cv_rmse_mean": float(-np.mean(rmse_scores)),
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "data_path": data_path,
        "model_path": str(model_path),
        "trained_at": ts,
    }
    meta_path = model_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return json.dumps({
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "model_type": model_type,
        "cv_r2": float(np.mean(r2_scores)),
        "cv_rmse": float(-np.mean(rmse_scores)),
        "n_samples": len(y),
        "n_features": len(feature_cols),
    }, default=str)


ML_TOOL_CALLABLES = {
    "run_model_comparison": run_model_comparison,
    "optimize_hyperparameters": optimize_hyperparameters,
    "train_final_model": train_final_model,
}
