"""
train_model.py - train the solubility model and save it for the app.

Usage:
    python train_model.py

Training data: ESOL (data/delaney-processed.csv) plus AqSolDB (Sorkun et al.,
2019), which is downloaded once and cached as data/aqsoldb.csv. AqSolDB adds
about 7,700 compounds, including about 170 with a P=O group, which is what lets
the nPO feature work (section 11 of molecules.ipynb in chem-ml-solubility).

Evaluation: every score is measured on ESOL molecules only, with the same folds
as chem-ml-solubility (5-fold CV x 3 shuffles, random and scaffold folds), so
the numbers are directly comparable with the original 0.709 / 0.804. AqSolDB is
only ever added to the training side of each fold:
  - every ESOL compound is removed from it first, matched on InChIKey
    connectivity so stereoisomers are caught too;
  - in scaffold folds, any AqSolDB compound sharing a ring framework with the
    test fold is also removed, so "unseen frameworks" stays true.
The final model is then fitted on all of ESOL + the cleaned AqSolDB.

Model: histogram gradient boosting. Its settings were chosen once, not tuned
against the test folds.
"""

import os
# Gradient boosting uses OpenMP threads. Using every core can make it crawl on
# WSL (threads waiting on each other), so cap it. Must be set before sklearn is
# imported. Override with e.g. OMP_NUM_THREADS=8 python train_model.py
os.environ.setdefault("OMP_NUM_THREADS", "4")
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from chem_utils import FEATURE_NAMES, MODEL_PATH, featurize, has_p_equals_o, parse_smiles

DATA_DIR = Path(__file__).parent / "data"
ESOL_CSV = DATA_DIR / "delaney-processed.csv"
AQSOLDB_CSV = DATA_DIR / "aqsoldb.csv"
AQSOLDB_URL = "https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv"

SEED = 42
CV_SEEDS = [SEED, SEED + 1, SEED + 2]  # three different shuffles of the folds


def make_model(seed=SEED):
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=10, l2_regularization=1.0, random_state=seed,
    )


# ------------------------------------------------------------------- data
def scaffold_of(mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def load_esol() -> pd.DataFrame:
    raw = pd.read_csv(ESOL_CSV)
    df = pd.DataFrame({
        "smiles": raw["smiles"].str.strip(),
        "logS": raw["measured log solubility in mols per litre"],
    })
    df["mol"] = df["smiles"].apply(parse_smiles)
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    df["inchikey"] = df["mol"].apply(Chem.MolToInchiKey)
    df["scaffold"] = df["mol"].apply(scaffold_of)
    return df


def load_aqsoldb(esol: pd.DataFrame) -> pd.DataFrame:
    """AqSolDB without salts/mixtures, duplicates or any ESOL compound."""
    if not AQSOLDB_CSV.exists():
        print("Downloading AqSolDB...")
        pd.read_csv(AQSOLDB_URL).to_csv(AQSOLDB_CSV, index=False)
    raw = pd.read_csv(AQSOLDB_CSV).rename(columns={"SMILES": "smiles", "Solubility": "logS"})
    raw = raw[~raw["smiles"].str.contains(".", regex=False)]  # drop salts and mixtures
    df = raw[["smiles", "logS"]].copy()
    df["mol"] = df["smiles"].apply(parse_smiles)
    df = df.dropna(subset=["mol"])
    df["inchikey"] = df["mol"].apply(Chem.MolToInchiKey)
    df = df[df["inchikey"] != ""].drop_duplicates(subset="inchikey")
    # remove ESOL compounds, including stereoisomers: the first 14 InChIKey
    # characters encode connectivity only
    esol_skeletons = set(esol["inchikey"].str[:14])
    df = df[~df["inchikey"].str[:14].isin(esol_skeletons)].reset_index(drop=True)
    df["scaffold"] = df["mol"].apply(scaffold_of)
    return df


def feature_matrix(mols, desc="Featurising") -> np.ndarray:
    rows = []
    for m in tqdm(mols, desc=desc, unit="mol"):
        feats = featurize(m)  # compute once per molecule, then pick the columns
        rows.append([feats[name] for name in FEATURE_NAMES])
    return np.array(rows)


# -------------------------------------------------------------- evaluation
def scaffold_folds(mols, n_splits=5, seed=SEED):
    """Split into folds so molecules sharing a ring framework stay in the same fold."""
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        groups[scaffold_of(m) or f"ring-free {i}"].append(i)  # ring-free molecules share no framework
    keys = list(groups)
    fold_of = np.empty(len(mols), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    for k in np.random.default_rng(seed).permutation(len(keys)):
        f = int(np.argmin(sizes))  # put each group in the currently smallest fold
        fold_of[groups[keys[k]]] = f
        sizes[f] += len(groups[keys[k]])
    idx = np.arange(len(mols))
    return [(idx[fold_of != f], idx[fold_of == f]) for f in range(n_splits)]


def cross_validate(esol, X_esol, aq, X_aq):
    """Scores on ESOL test folds, with AqSolDB added to each training fold."""
    y = esol["logS"].to_numpy()
    y_aq = aq["logS"].to_numpy()
    is_po = esol["mol"].apply(has_p_equals_o).to_numpy()
    results = {}
    for split in ["random", "scaffold"]:
        rmses, r2s, po_maes = [], [], []
        for seed in tqdm(CV_SEEDS, desc=f"CV, {split} folds", unit="shuffle"):
            if split == "random":
                folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_esol))
            else:
                folds = scaffold_folds(esol["mol"].tolist(), seed=seed)
            pred = np.empty(len(y))
            for train_idx, test_idx in folds:
                keep = np.ones(len(aq), dtype=bool)
                if split == "scaffold":  # no AqSolDB compound sharing a test-fold framework
                    test_scaffolds = {s for s in esol["scaffold"].iloc[test_idx] if s}
                    keep = ~aq["scaffold"].isin(test_scaffolds).to_numpy()
                X_train = np.vstack([X_esol[train_idx], X_aq[keep]])
                y_train = np.concatenate([y[train_idx], y_aq[keep]])
                pred[test_idx] = make_model(seed).fit(X_train, y_train).predict(X_esol[test_idx])
            err = y - pred
            rmses.append(np.sqrt(mean_squared_error(y, pred)))
            r2s.append(r2_score(y, pred))
            po_maes.append(np.abs(err[is_po]).mean())
        results[split] = {
            "RMSE": float(np.mean(rmses)),
            "RMSE_sd": float(np.std(rmses, ddof=1)),
            "R2": float(np.mean(r2s)),
            "MAE_PO": float(np.mean(po_maes)),
        }
        print(f"{split:8s} folds:  RMSE = {results[split]['RMSE']:.3f} "
              f"(sd {results[split]['RMSE_sd']:.3f})   R^2 = {results[split]['R2']:.3f}   "
              f"MAE on {int(is_po.sum())} P=O compounds = {results[split]['MAE_PO']:.2f}")
    return results


# -------------------------------------------------------------------- main
def main():
    esol = load_esol()
    aq = load_aqsoldb(esol)
    print(f"ESOL: {len(esol)} molecules   AqSolDB after cleaning: {len(aq)} molecules")

    X_esol = feature_matrix(esol["mol"], desc="Featurising ESOL")
    X_aq = feature_matrix(aq["mol"], desc="Featurising AqSolDB")

    print("Cross-validating on ESOL (5 folds x 3 shuffles, random and scaffold)...")
    cv = cross_validate(esol, X_esol, aq, X_aq)

    # final model uses every molecule
    print("Training the final model and measuring feature importances...")
    X = np.vstack([X_esol, X_aq])
    y = np.concatenate([esol["logS"].to_numpy(), aq["logS"].to_numpy()])
    model = make_model().fit(X, y)

    # gradient boosting has no built-in importances like a random forest, so
    # measure how much the error grows when each feature is shuffled, then
    # scale so the importances add up to 1 for display
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=SEED, n_jobs=-1)
    raw = np.clip(perm.importances_mean, 0, None)
    importances = dict(zip(FEATURE_NAMES, (raw / raw.sum()).round(3).tolist()))
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
    print("Top features:", top)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURE_NAMES, "importances": importances,
         "cv": cv, "n_train": len(y)},
        MODEL_PATH,
    )
    print(f"Saved model ({len(y)} training molecules) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
