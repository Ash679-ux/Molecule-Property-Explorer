"""Chemistry and model code for the Molecule Property Explorer.

Everything that uses RDKit or the model lives here; app.py only handles the interface.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")  # invalid SMILES are reported by the app instead

MODEL_PATH = Path(__file__).parent / "models" / "solubility_model.joblib"

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
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
FEATURE_NAMES = list(FEATURE_LABELS)

# Datasets write P=O either as a double bond or as charge-separated P+-O-.
PHOSPHORYL = [Chem.MolFromSmarts("[#15]=[#8]"), Chem.MolFromSmarts("[#15+]-[#8-]")]
THIOPHOSPHORYL = [Chem.MolFromSmarts("[#15]=[#16]"), Chem.MolFromSmarts("[#15+]-[#16-]")]

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DOMAIN_THRESHOLD = 0.3  # below this similarity the cross-validated error rises clearly
DOMAIN_BANDS = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0)]


# Parsing and features

def parse_smiles(smiles):
    """RDKit molecule, or None if the SMILES is empty or invalid."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical_smiles(mol):
    return Chem.MolToSmiles(mol)


def count_matches(mol, patterns):
    return sum(len(mol.GetSubstructMatches(p)) for p in patterns)


def has_phosphoryl(mol):
    """True for phosphates and phosphonates, historically the model's hardest class."""
    return count_matches(mol, PHOSPHORYL) > 0


def aromatic_proportion(mol):
    heavy = mol.GetNumHeavyAtoms()
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / heavy if heavy else 0.0


def element_counts(mol):
    counts = dict.fromkeys(ELEMENTS, 0)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in counts:
            counts[atom.GetSymbol()] += 1
    return {f"n{el}": n for el, n in counts.items()}


def featurize(mol):
    """The model's 25 descriptors, in FEATURE_NAMES order."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "AromProp": aromatic_proportion(mol),
        "nPO": count_matches(mol, PHOSPHORYL),
        "nPS": count_matches(mol, THIOPHOSPHORYL),
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "AromRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "MolMR": Crippen.MolMR(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        **element_counts(mol),
    }


def feature_vector(mol):
    features = featurize(mol)
    return np.array([[features[name] for name in FEATURE_NAMES]])


# Model

def load_model(path=MODEL_PATH):
    """The bundle saved by train_model.py: the model plus its evaluation results."""
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Run `python train_model.py` first.")
    bundle = joblib.load(path)
    if bundle["features"] != FEATURE_NAMES:
        raise ValueError("The saved model uses different features. Run `python train_model.py`.")
    return bundle


def predict_logS(mol, bundle):
    """Predicted log10 solubility in mol/L."""
    return float(bundle["model"].predict(feature_vector(mol))[0])


def solubility_label(logS):
    """Rough plain-English band, for readability only."""
    if logS >= 0:
        return "Highly soluble"
    if logS >= -2:
        return "Soluble"
    if logS >= -4:
        return "Moderately soluble"
    if logS >= -6:
        return "Poorly soluble"
    return "Practically insoluble"


def logS_to_mg_per_ml(logS, mol):
    """mol/L x g/mol = g/L, which equals mg/mL."""
    return 10 ** logS * Descriptors.MolWt(mol)


# Applicability domain

def fingerprint(mol):
    return MORGAN.GetFingerprint(mol)


def nearest_neighbour(mol, train_fps):
    """(Tanimoto similarity, index) of the most similar training molecule."""
    sims = DataStructs.BulkTanimotoSimilarity(fingerprint(mol), train_fps)
    idx = int(np.argmax(sims))
    return sims[idx], idx


def band_of(similarity):
    for lo, hi in DOMAIN_BANDS:
        if lo <= similarity < hi:
            return lo, hi
    return DOMAIN_BANDS[-1]  # similarity of exactly 1.0


# Why view

def logp_contributions(mol):
    """Each heavy atom's share of Crippen logP, with its hydrogens' shares added in.

    Sums exactly to Crippen.MolLogP(mol) (Wildman & Crippen, 1999).
    """
    with_h = Chem.AddHs(mol)  # heavy atoms keep their indices
    contribs = [c[0] for c in rdMolDescriptors._CalcCrippenContribs(with_h)]
    weights = [0.0] * mol.GetNumAtoms()
    for atom in with_h.GetAtoms():
        owner = atom.GetNeighbors()[0] if atom.GetAtomicNum() == 1 else atom
        weights[owner.GetIdx()] += contribs[atom.GetIdx()]
    return weights


def logp_heatmap_png(mol, size=400):
    """Molecule shaded red where atoms raise logP and blue where they lower it."""
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, logp_contributions(mol), draw2d=drawer, colorMap="bwr"
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# Drug-likeness

def rule_of_five(mol):
    """Lipinski's rule of five with the original counts: NH + OH donors, N + O acceptors."""
    rules = [
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


def qed_score(mol):
    """Quantitative Estimate of Drug-likeness (Bickerton et al., 2012), from 0 to 1."""
    return QED.qed(mol)


# Structural alerts

def build_alert_catalogs():
    """RDKit's PAINS (480 patterns) and Brenk (105 patterns) catalogs."""
    catalogs = {}
    for name, which in [("PAINS", FilterCatalogParams.FilterCatalogs.PAINS),
                        ("Brenk", FilterCatalogParams.FilterCatalogs.BRENK)]:
        params = FilterCatalogParams()
        params.AddCatalog(which)
        catalogs[name] = FilterCatalog(params)
    return catalogs


def structural_alerts(mol, catalogs):
    """Each matching alert as {"name", "source", "atoms"} (all places it matches)."""
    alerts = []
    for source, catalog in catalogs.items():
        for entry in catalog.GetMatches(mol):
            atoms = {pair.target for match in entry.GetFilterMatches(mol) for pair in match.atomPairs}
            # PAINS names end in their hit count from the original screen, e.g. "ene_rhod_A(235)"
            name = re.sub(r"\(\d+\)$", "", entry.GetDescription()).replace("_", " ")
            alerts.append({"name": name, "source": source, "atoms": sorted(atoms)})
    return alerts


def bonds_between(mol, atoms):
    atoms = set(atoms)
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]


# Batch mode

def predict_batch(smiles_list, bundle, catalogs, train_fps):
    """One row of results per SMILES; invalid SMILES are kept and marked invalid."""
    rows = []
    for smiles in smiles_list:
        smiles = "" if pd.isna(smiles) else str(smiles)
        mol = parse_smiles(smiles)
        if mol is None:
            rows.append({"smiles": smiles, "valid": False})
            continue
        logS = predict_logS(mol, bundle)
        similarity, _ = nearest_neighbour(mol, train_fps)
        rows.append({
            "smiles": smiles,
            "valid": True,
            "logS": round(logS, 2),
            "mg_per_mL": float(f"{logS_to_mg_per_ml(logS, mol):.3g}"),
            "solubility": solubility_label(logS),
            "QED": round(qed_score(mol), 2),
            "Ro5_violations": rule_of_five(mol)["violations"],
            "alerts": "; ".join(a["name"] for a in structural_alerts(mol, catalogs)),
            "nearest_similarity": round(similarity, 2),
            "outside_domain": similarity < DOMAIN_THRESHOLD,
        })
    return pd.DataFrame(rows)
