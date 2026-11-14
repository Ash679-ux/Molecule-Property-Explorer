"""Tests for chem_utils.py. Run with: pytest"""

import math

import pytest
from rdkit.Chem import Crippen

import chem_utils as cu

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ERYTHROMYCIN = (
    "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)"
    "C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O"
)
RHODANINE = "O=C1NC(=S)SC1=Cc1ccccc1"


@pytest.fixture(scope="module")
def bundle():
    return cu.load_model()


@pytest.fixture(scope="module")
def catalogs():
    return cu.build_alert_catalogs()


@pytest.fixture(scope="module")
def train_fps(bundle):
    return [cu.fingerprint(cu.parse_smiles(s)) for s in bundle["train"]["smiles"]]


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("bad", ["", "   ", "C1CC(", "this_is_not_smiles", None])
def test_invalid_smiles_returns_none(bad):
    assert cu.parse_smiles(bad) is None


def test_whitespace_is_stripped():
    assert cu.parse_smiles("  CCO  ") is not None


# --------------------------------------------------------------- features
def test_featurize_matches_feature_names():
    assert list(cu.featurize(cu.parse_smiles(ASPIRIN))) == cu.FEATURE_NAMES
    assert set(cu.FEATURE_LABELS) == set(cu.FEATURE_NAMES)


@pytest.mark.parametrize("smiles", ["CCOP(=O)(OCC)OCC", "CCO[P+]([O-])(OCC)OCC"])
def test_npo_counts_both_encodings(smiles):
    """P=O written as a double bond or as charge-separated P+-O- counts the same."""
    assert cu.featurize(cu.parse_smiles(smiles))["nPO"] == 1


def test_thiophosphate_is_ps_not_po():
    feats = cu.featurize(cu.parse_smiles("CCOP(=S)(OCC)OCC"))
    assert feats["nPS"] == 1 and feats["nPO"] == 0


# ------------------------------------------------------------------ model
def test_model_matches_features(bundle):
    assert bundle["features"] == cu.FEATURE_NAMES


def test_prediction_is_plausible(bundle):
    logS = cu.predict_logS(cu.parse_smiles(ASPIRIN), bundle)
    assert math.isfinite(logS) and -4 < logS < 0  # aspirin is about -1.7 measured


def test_mg_per_ml_conversion():
    mol = cu.parse_smiles(ASPIRIN)  # MW 180.16
    assert cu.logS_to_mg_per_ml(0.0, mol) == pytest.approx(180.16, abs=0.01)


# ---------------------------------------------------------- drug-likeness
def test_aspirin_passes_rule_of_five():
    assert cu.lipinski_report(cu.parse_smiles(ASPIRIN))["drug_like"]


def test_erythromycin_fails_rule_of_five():
    report = cu.lipinski_report(cu.parse_smiles(ERYTHROMYCIN))
    assert report["violations"] >= 2 and not report["drug_like"]


def test_qed_in_range():
    assert 0 <= cu.qed_score(cu.parse_smiles(ASPIRIN)) <= 1


# ------------------------------------------------------- structural alerts
def test_rhodanine_triggers_pains(catalogs):
    alerts = cu.structural_alerts(cu.parse_smiles(RHODANINE), catalogs)
    assert any(a["source"] == "PAINS" for a in alerts)


def test_benzene_has_no_alerts(catalogs):
    assert cu.structural_alerts(cu.parse_smiles("c1ccccc1"), catalogs) == []


def test_alert_atoms_are_valid_indices(catalogs):
    mol = cu.parse_smiles(RHODANINE)
    for alert in cu.structural_alerts(mol, catalogs):
        assert alert["atoms"] and all(0 <= i < mol.GetNumAtoms() for i in alert["atoms"])


# ---------------------------------------------------------------- why view
@pytest.mark.parametrize("smiles", [ASPIRIN, "OCC1OC(O)C(O)C(O)C1O", "CCOP(=O)(OCC)OCC"])
def test_logp_contributions_sum_to_logp(smiles):
    mol = cu.parse_smiles(smiles)
    assert sum(cu.logp_contributions(mol)) == pytest.approx(Crippen.MolLogP(mol), abs=1e-6)


def test_heatmap_is_png():
    assert cu.logp_heatmap_png(cu.parse_smiles(ASPIRIN))[:4] == b"\x89PNG"


# ---------------------------------------------------- applicability domain
def test_training_molecule_is_its_own_neighbour(train_fps):
    sim, _ = cu.nearest_neighbour(cu.parse_smiles("c1ccccc1"), train_fps)
    assert sim == pytest.approx(1.0)


def test_salt_is_outside_domain(train_fps):
    sim, _ = cu.nearest_neighbour(cu.parse_smiles("[Na+].[Cl-]"), train_fps)
    assert sim < cu.DOMAIN_THRESHOLD


def test_bands_cover_zero_to_one():
    assert cu.band_of(0.0) == cu.DOMAIN_BANDS[0]
    assert cu.band_of(1.0) == cu.DOMAIN_BANDS[-1]


# --------------------------------------------------------------- batch mode
def test_predict_table_handles_invalid_rows(bundle, catalogs, train_fps):
    table = cu.predict_table([ASPIRIN, "not smiles", None], bundle, catalogs, train_fps)
    assert list(table["valid"]) == [True, False, False]
    assert table.loc[0, "Ro5_violations"] == 0
