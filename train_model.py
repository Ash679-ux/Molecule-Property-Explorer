"""
train_model.py - train the ESOL solubility model and save it for the app.

Usage:
    python train_model.py                      # uses data/delaney-processed.csv
    python train_model.py path/to/esol.csv     # or point it at your own copy

The CSV needs a 'smiles' column and a
'measured log solubility in mols per litre' column (the MoleculeNet ESOL file).

Evaluation follows my chem-ml-solubility analysis (molecules.ipynb):
5-fold cross-validation repeated with three shuffles, once with random folds
and once with scaffold folds (molecules sharing a Murcko ring framework stay in
the same fold). The final model is then fitted on all molecules; the
cross-validated scores are what describe how well it does on new ones.
"""

import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from chem_utils import FEATURE_NAMES, MODEL_PATH, featurize, parse_smiles

TARGET_COL = "measured log solubility in mols per litre"
SMILES_COL = "smiles"
SEED = 42
CV_SEEDS = [SEED, SEED + 1, SEED + 2]  # three different shuffles of the folds


def make_model(seed=SEED):
    return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)


def build_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    rows, targets, mols, skipped = [], [], [], 0
    for smi, y in zip(df[SMILES_COL], df[TARGET_COL]):
        mol = parse_smiles(smi)
        if mol is None:
            skipped += 1
            continue
        feats = featurize(mol)
        rows.append([feats[name] for name in FEATURE_NAMES])
        targets.append(y)
        mols.append(mol)
    print(f"Loaded {len(rows)} molecules ({skipped} skipped as unparseable)")
    return np.array(rows), np.array(targets), mols


def scaffold_folds(mols, n_splits=5, seed=SEED):
    """Split into folds so molecules sharing a ring framework stay in the same fold."""
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        groups[scaffold or f"ring-free {i}"].append(i)  # ring-free molecules share no framework
    keys = list(groups)
    fold_of = np.empty(len(mols), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    for k in np.random.default_rng(seed).permutation(len(keys)):
        f = int(np.argmin(sizes))  # put each group in the currently smallest fold
        fold_of[groups[keys[k]]] = f
        sizes[f] += len(groups[keys[k]])
    idx = np.arange(len(mols))
    return [(idx[fold_of != f], idx[fold_of == f]) for f in range(n_splits)]


def cross_validate(X, y, mols):
    """Mean and SD of RMSE and R2 over three shuffles, for random and scaffold folds."""
    results = {}
    for split in ["random", "scaffold"]:
        rmses, r2s = [], []
        for seed in CV_SEEDS:
            if split == "random":
                folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))
            else:
                folds = scaffold_folds(mols, seed=seed)
            pred = cross_val_predict(make_model(seed), X, y, cv=folds)  # every molecule predicted once
            rmses.append(np.sqrt(mean_squared_error(y, pred)))
            r2s.append(r2_score(y, pred))
        results[split] = {
            "RMSE": float(np.mean(rmses)),
            "RMSE_sd": float(np.std(rmses, ddof=1)),
            "R2": float(np.mean(r2s)),
        }
        print(f"{split:8s} folds:  RMSE = {results[split]['RMSE']:.3f} "
              f"(sd {results[split]['RMSE_sd']:.3f})   R^2 = {results[split]['R2']:.3f}")
    return results


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/delaney-processed.csv")
    X, y, mols = build_dataset(csv_path)

    print("Cross-validating (5 folds x 3 shuffles, random and scaffold)...")
    cv = cross_validate(X, y, mols)

    model = make_model().fit(X, y)  # final model uses every molecule
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.round(3).tolist()))
    print("Feature importances:", importances)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURE_NAMES, "importances": importances, "cv": cv},
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
