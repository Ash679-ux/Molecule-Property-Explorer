"""
chem_utils.py - the chemistry "engine" behind the app.

Everything RDKit/ML lives here, so app.py only deals with the interface.
That separation makes this file easy to test on its own:

    python chem_utils.py
"""

from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# RDKit prints noisy warnings for bad SMILES; we handle errors ourselves.
RDLogger.DisableLog("rdApp.*")

MODEL_PATH = Path(__file__).parent / "models" / "esol_model.joblib"

# The four descriptors from Delaney's original ESOL paper (2004).
# If your chem-ml-solubility repo uses different features, change
# FEATURE_NAMES and featurize() together, then retrain.
FEATURE_NAMES = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]


# ---------------------------------------------------------------- parsing
def parse_smiles(smiles: str):
    """Return an RDKit Mol, or None if the SMILES is empty/invalid."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical_smiles(mol) -> str:
    return Chem.MolToSmiles(mol)


# ------------------------------------------------------------- featurising
def aromatic_proportion(mol) -> float:
    """Fraction of heavy atoms that are aromatic."""
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return aromatic / heavy


def featurize(mol) -> dict:
    """Turn a molecule into the numbers the model was trained on."""
    return {
        "MolLogP": Descriptors.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticProportion": aromatic_proportion(mol),
    }


def feature_vector(mol) -> np.ndarray:
    feats = featurize(mol)
    return np.array([[feats[name] for name in FEATURE_NAMES]])


# ------------------------------------------------------------------ model
def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the bundle saved by train_model.py.

    The bundle is a dict: {"model", "features", "test_rmse", "test_r2"}.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. Run `python train_model.py` first."
        )
    bundle = joblib.load(path)
    if bundle["features"] != FEATURE_NAMES:
        raise ValueError(
            "Saved model was trained on different features than featurize() "
            "produces. Retrain with `python train_model.py`."
        )
    return bundle


def predict_logS(mol, bundle: dict) -> float:
    """Predicted log10 solubility in mol/L."""
    return float(bundle["model"].predict(feature_vector(mol))[0])


# ------------------------------------------------------- interpretation
def solubility_label(logS: float) -> str:
    """Rough plain-English band for a logS value (mol/L).

    These cut-offs are approximate and are for readability only.
    """
    if logS >= 0:
        return "Highly soluble"
    if logS >= -2:
        return "Soluble"
    if logS >= -4:
        return "Moderately soluble"
    if logS >= -6:
        return "Poorly soluble"
    return "Practically insoluble"


def logS_to_mg_per_ml(logS: float, mol) -> float:
    """Convert logS (mol/L) to mg/mL: (mol/L) x (g/mol) = g/L = mg/mL."""
    return (10 ** logS) * Descriptors.MolWt(mol)


# ------------------------------------------------------------ smoke test
if __name__ == "__main__":
    bundle = load_model()
    tests = {
        "benzene": "c1ccccc1",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "glucose": "OCC1OC(O)C(O)C(O)C1O",
        "not a molecule": "this_is_not_smiles",
    }
    for name, smi in tests.items():
        mol = parse_smiles(smi)
        if mol is None:
            print(f"{name:15s}  invalid SMILES (correctly rejected)")
            continue
        logS = predict_logS(mol, bundle)
        print(f"{name:15s}  logS = {logS:6.2f}  ({solubility_label(logS)})")
