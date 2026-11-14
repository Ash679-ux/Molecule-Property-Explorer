"""Train the solubility model and save it for the app. Run with: python train_model.py

Training data is ESOL plus AqSolDB (downloaded on first run). Every score is measured
on ESOL with the same folds as chem-ml-solubility (5 folds x 3 shuffles, random and
scaffold), so results compare directly with its 0.709 / 0.804. AqSolDB only ever joins
the training folds: ESOL compounds are removed from it (including stereoisomers), and
in scaffold folds so is anything sharing a test molecule's ring framework.
The final model is then fitted on everything.
"""

import os

# Using every core can stall gradient boosting on WSL. Must be set before importing sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "4")

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

import chem_utils as cu

DATA_DIR = Path(__file__).parent / "data"
ESOL_CSV = DATA_DIR / "delaney-processed.csv"
AQSOLDB_CSV = DATA_DIR / "aqsoldb.csv"
AQSOLDB_URL = "https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv"

SEED = 42
CV_SEEDS = [SEED, SEED + 1, SEED + 2]


def make_model(seed=SEED):
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=10, l2_regularization=1.0, random_state=seed,
    )


# Data

def scaffold_of(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def load_esol():
    raw = pd.read_csv(ESOL_CSV)
    df = pd.DataFrame({
        "smiles": raw["smiles"].str.strip(),
        "logS": raw["measured log solubility in mols per litre"],
    })
    df["mol"] = df["smiles"].apply(cu.parse_smiles)
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    df["inchikey"] = df["mol"].apply(Chem.MolToInchiKey)
    df["scaffold"] = df["mol"].apply(scaffold_of)
    return df


def load_aqsoldb(esol):
    """AqSolDB without salts, mixtures, duplicates or any ESOL compound."""
    if not AQSOLDB_CSV.exists():
        print("Downloading AqSolDB...")
        pd.read_csv(AQSOLDB_URL).to_csv(AQSOLDB_CSV, index=False)
    raw = pd.read_csv(AQSOLDB_CSV).rename(columns={"SMILES": "smiles", "Solubility": "logS"})
    df = raw.loc[~raw["smiles"].str.contains(".", regex=False), ["smiles", "logS"]].copy()
    df["mol"] = df["smiles"].apply(cu.parse_smiles)
    df = df.dropna(subset=["mol"])
    df["inchikey"] = df["mol"].apply(Chem.MolToInchiKey)
    df = df[df["inchikey"] != ""].drop_duplicates(subset="inchikey")
    # The first 14 InChIKey characters encode connectivity only, so this also catches stereoisomers.
    esol_skeletons = set(esol["inchikey"].str[:14])
    df = df[~df["inchikey"].str[:14].isin(esol_skeletons)].reset_index(drop=True)
    df["scaffold"] = df["mol"].apply(scaffold_of)
    return df


def feature_matrix(mols, desc):
    return np.array([cu.feature_vector(m)[0] for m in tqdm(mols, desc=desc, unit="mol")])


# Evaluation

def scaffold_folds(mols, n_splits=5, seed=SEED):
    """Folds in which molecules sharing a ring framework always stay together."""
    groups = defaultdict(list)
    for i, mol in enumerate(mols):
        groups[scaffold_of(mol) or f"ring-free {i}"].append(i)
    keys = list(groups)
    fold_of = np.empty(len(mols), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    for k in np.random.default_rng(seed).permutation(len(keys)):
        fold = int(np.argmin(sizes))
        fold_of[groups[keys[k]]] = fold
        sizes[fold] += len(groups[keys[k]])
    idx = np.arange(len(mols))
    return [(idx[fold_of != f], idx[fold_of == f]) for f in range(n_splits)]


def cross_validate(esol, X_esol, aq, X_aq):
    """Scores on ESOL test folds, plus random-fold error by similarity to the training fold."""
    y, y_aq = esol["logS"].to_numpy(), aq["logS"].to_numpy()
    is_phosphoryl = esol["mol"].apply(cu.has_phosphoryl).to_numpy()
    fps_esol = [cu.fingerprint(m) for m in esol["mol"]]
    fps_aq = [cu.fingerprint(m) for m in aq["mol"]]
    similarities, errors = [], []
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
                if split == "scaffold":
                    test_scaffolds = {s for s in esol["scaffold"].iloc[test_idx] if s}
                    keep = ~aq["scaffold"].isin(test_scaffolds).to_numpy()
                X_train = np.vstack([X_esol[train_idx], X_aq[keep]])
                y_train = np.concatenate([y[train_idx], y_aq[keep]])
                pred[test_idx] = make_model(seed).fit(X_train, y_train).predict(X_esol[test_idx])

                if split == "random":
                    train_fps = [fps_esol[i] for i in train_idx] + fps_aq
                    for i in test_idx:
                        similarities.append(max(DataStructs.BulkTanimotoSimilarity(fps_esol[i], train_fps)))
                        errors.append(y[i] - pred[i])

            rmses.append(np.sqrt(mean_squared_error(y, pred)))
            r2s.append(r2_score(y, pred))
            po_maes.append(np.abs(y - pred)[is_phosphoryl].mean())

        results[split] = {
            "RMSE": float(np.mean(rmses)),
            "RMSE_sd": float(np.std(rmses, ddof=1)),
            "R2": float(np.mean(r2s)),
            "MAE_PO": float(np.mean(po_maes)),
        }
        r = results[split]
        print(f"{split:8s} folds: RMSE = {r['RMSE']:.3f} (sd {r['RMSE_sd']:.3f}), R^2 = {r['R2']:.3f}, "
              f"MAE on {int(is_phosphoryl.sum())} P=O compounds = {r['MAE_PO']:.2f}")

    return results, error_by_similarity(np.array(similarities), np.array(errors))


def error_by_similarity(similarities, errors):
    domain = []
    bands = np.array([cu.band_of(s) for s in similarities])
    for lo, hi in cu.DOMAIN_BANDS:
        in_band = (bands[:, 0] == lo) & (bands[:, 1] == hi)
        n = int(in_band.sum()) // len(CV_SEEDS)  # molecules, not predictions
        rmse = float(np.sqrt((errors[in_band] ** 2).mean())) if in_band.any() else None
        domain.append({"lo": lo, "hi": hi, "n": n, "RMSE": rmse})
        print(f"  similarity {lo:.1f}-{hi:.1f}: {n:4d} molecules, RMSE = {rmse:.2f}")
    return domain


# Training

def main():
    esol = load_esol()
    aq = load_aqsoldb(esol)
    print(f"ESOL: {len(esol)} molecules, AqSolDB after cleaning: {len(aq)} molecules")

    X_esol = feature_matrix(esol["mol"], "Featurising ESOL")
    X_aq = feature_matrix(aq["mol"], "Featurising AqSolDB")

    print("Cross-validating on ESOL...")
    cv, domain = cross_validate(esol, X_esol, aq, X_aq)

    print("Training the final model and measuring feature importances...")
    X = np.vstack([X_esol, X_aq])
    y = np.concatenate([esol["logS"].to_numpy(), aq["logS"].to_numpy()])
    model = make_model().fit(X, y)

    # Gradient boosting has no built-in importances, so measure how much shuffling each feature hurts.
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=SEED, n_jobs=-1)
    raw = np.clip(perm.importances_mean, 0, None)
    importances = dict(zip(cu.FEATURE_NAMES, (raw / raw.sum()).round(3).tolist()))
    print("Top features:", sorted(importances.items(), key=lambda kv: -kv[1])[:5])

    mols = list(esol["mol"]) + list(aq["mol"])
    bundle = {
        "model": model,
        "features": cu.FEATURE_NAMES,
        "importances": importances,
        "cv": cv,
        "domain": domain,
        "n_train": len(y),
        "train": {
            "smiles": [cu.canonical_smiles(m) for m in mols],
            "logS": y.tolist(),
            "source": ["ESOL"] * len(esol) + ["AqSolDB"] * len(aq),
        },
    }
    cu.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cu.MODEL_PATH)
    print(f"Saved model ({len(y)} training molecules) to {cu.MODEL_PATH}")


if __name__ == "__main__":
    main()
