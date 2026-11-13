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
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

# RDKit prints noisy warnings for bad SMILES; we handle errors ourselves.
RDLogger.DisableLog("rdApp.*")

MODEL_PATH = Path(__file__).parent / "models" / "esol_model.joblib"

# Features: the seven descriptors from my chem-ml-solubility analysis
# (molecules.ipynb), plus nPO (the number of P=O groups, from section 11 of
# that notebook), plus extra descriptors and element counts that lowered RMSE
# in benchmarking. Change FEATURE_NAMES and featurize() together, then retrain.
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]  # every element in ESOL
FEATURE_NAMES = (
    ["MolWt", "LogP", "HBD", "HBA", "RotBonds", "TPSA", "AromProp", "nPO",
     "nPS", "FracCSP3", "AromRings", "Rings", "MolMR", "HeavyAtoms", "LabuteASA", "BertzCT"]
    + [f"n{el}" for el in ELEMENTS]
)

# Plain-English names for the app's descriptor table.
FEATURE_LABELS = {
    "MolWt": "Molecular weight (g/mol)",
    "LogP": "logP (Crippen)",
    "HBD": "H-bond donors",
    "HBA": "H-bond acceptors",
    "RotBonds": "Rotatable bonds",
    "TPSA": "Polar surface area (Å²)",
    "AromProp": "Aromatic proportion",
    "nPO": "P=O groups",
    "nPS": "P=S groups",
    "FracCSP3": "Fraction of sp³ carbons",
    "AromRings": "Aromatic rings",
    "Rings": "Rings",
    "MolMR": "Molar refractivity (Crippen)",
    "HeavyAtoms": "Heavy atoms",
    "LabuteASA": "Approx. surface area (Labute, Å²)",
    "BertzCT": "Structural complexity (Bertz)",
    **{f"n{el}": f"{el} atoms" for el in ELEMENTS},
}

# P=O and P=S groups, however the SMILES draws them: as a double bond, or as
# P+ single-bonded to O- / S- (some entries in AqSolDB use this form).
# Same patterns as section 11 of molecules.ipynb.
PO_PATTERNS = [Chem.MolFromSmarts("[#15]=[#8]"), Chem.MolFromSmarts("[#15+]-[#8-]")]
PS_PATTERNS = [Chem.MolFromSmarts("[#15]=[#16]"), Chem.MolFromSmarts("[#15+]-[#16-]")]


def count_matches(mol, patterns) -> int:
    """Total number of matches of any of the SMARTS patterns in mol."""
    return sum(len(mol.GetSubstructMatches(p)) for p in patterns)


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


def element_counts(mol) -> dict:
    counts = dict.fromkeys(ELEMENTS, 0)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in counts:
            counts[atom.GetSymbol()] += 1
    return {f"n{el}": n for el, n in counts.items()}


def featurize(mol) -> dict:
    """Turn a molecule into the numbers the model was trained on."""
    return {
        # the seven descriptors from chem-ml-solubility
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "AromProp": aromatic_proportion(mol),
        # the P=O count that fixed most of the phosphate/phosphonate error
        "nPO": count_matches(mol, PO_PATTERNS),
        # extras
        "nPS": count_matches(mol, PS_PATTERNS),
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "AromRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "MolMR": Crippen.MolMR(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        **element_counts(mol),
    }


def has_p_equals_o(mol) -> bool:
    """True for phosphates/phosphonates, historically this model's hardest class."""
    return count_matches(mol, PO_PATTERNS) > 0


# ---------------------------------------------------------- drug-likeness
def lipinski_report(mol) -> dict:
    """Lipinski's rule of five (Lipinski et al., 1997).

    Uses the rule's original definitions: H-bond donors = number of NH and OH
    groups, H-bond acceptors = number of N and O atoms. These differ from the
    HBD/HBA descriptors the solubility model uses (RDKit's more chemically
    careful counts), e.g. caffeine has 6 N + O but only 3 RDKit acceptors.
    A molecule is usually called drug-like (orally) with at most one violation.
    """
    rules = [
        # (property, value, limit)
        ("Molecular weight (g/mol)", Descriptors.MolWt(mol), 500),
        ("logP (Crippen)", Crippen.MolLogP(mol), 5),
        ("H-bond donors (NH + OH)", Lipinski.NHOHCount(mol), 5),
        ("H-bond acceptors (N + O)", Lipinski.NOCount(mol), 10),
    ]
    rows = [
        {"Property": name, "Value": value, "Limit": f"≤ {limit}", "Passes": value <= limit}
        for name, value, limit in rules
    ]
    violations = sum(not row["Passes"] for row in rows)
    return {"rows": rows, "violations": violations, "drug_like": violations <= 1}


def qed_score(mol) -> float:
    """Quantitative Estimate of Drug-likeness (Bickerton et al., 2012), 0 to 1.

    Combines eight properties (MW, logP, H-bond donors/acceptors, polar surface
    area, rotatable bonds, aromatic rings, structural alerts) into one score.
    Higher is more drug-like.
    """
    return QED.qed(mol)


def feature_vector(mol) -> np.ndarray:
    feats = featurize(mol)
    return np.array([[feats[name] for name in FEATURE_NAMES]])


# ------------------------------------------------------------------ model
def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the bundle saved by train_model.py.

    The bundle is a dict with keys "model", "features", "importances", "cv"
    (cross-validated RMSE/R2/P=O MAE for random and scaffold folds) and
    "n_train" (molecules in the final training set); see train_model.py.
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
        "triethyl phosphate": "CCOP(=O)(OCC)OCC",
        "not a molecule": "this_is_not_smiles",
    }
    for name, smi in tests.items():
        mol = parse_smiles(smi)
        if mol is None:
            print(f"{name:20s}  invalid SMILES (correctly rejected)")
            continue
        logS = predict_logS(mol, bundle)
        flag = "  [P=O warning]" if has_p_equals_o(mol) else ""
        lip = lipinski_report(mol)
        print(f"{name:20s}  logS = {logS:6.2f}  ({solubility_label(logS)})  "
              f"QED = {qed_score(mol):.2f}  Ro5 violations = {lip['violations']}{flag}")
