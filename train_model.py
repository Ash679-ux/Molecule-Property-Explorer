"""
train_model.py - train the ESOL solubility model and save it for the app.

Usage:
    python train_model.py                      # uses data/delaney-processed.csv
    python train_model.py path/to/esol.csv     # or point it at your own copy

The CSV needs a 'smiles' column and a
'measured log solubility in mols per litre' column (the MoleculeNet ESOL file).
You probably already have this file in your chem-ml-solubility repo.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chem_utils import FEATURE_NAMES, MODEL_PATH, featurize, parse_smiles

TARGET_COL = "measured log solubility in mols per litre"
SMILES_COL = "smiles"


def build_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    rows, targets, skipped = [], [], 0
    for smi, y in zip(df[SMILES_COL], df[TARGET_COL]):
        mol = parse_smiles(smi)
        if mol is None:
            skipped += 1
            continue
        feats = featurize(mol)
        rows.append([feats[name] for name in FEATURE_NAMES])
        targets.append(y)
    print(f"Loaded {len(rows)} molecules ({skipped} skipped as unparseable)")
    return np.array(rows), np.array(targets)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/delaney-processed.csv")
    X, y = build_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    print(f"Random-split test set:  RMSE = {rmse:.3f} log units   R^2 = {r2:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURE_NAMES, "test_rmse": rmse, "test_r2": r2},
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
