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
    assert cu.rule_of_five(cu.parse_smiles(ASPIRIN))["drug_like"]


def test_erythromycin_fails_rule_of_five():
    report = cu.rule_of_five(cu.parse_smiles(ERYTHROMYCIN))
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
def test_predict_batch_handles_invalid_rows(bundle, catalogs, train_fps):
    table = cu.predict_batch([ASPIRIN, "not smiles", None], bundle, catalogs, train_fps)
    assert list(table["valid"]) == [True, False, False]
    assert table.loc[0, "Ro5_violations"] == 0


# ------------------------------------------------------------- input checks
@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_has_no_problem_message(text):
    assert cu.check_smiles(text) == (None, None)


@pytest.mark.parametrize("text, expected", [
    ("CCO CCC", "spaces"),
    ("CC(C", "brackets"),
    ("C[CH2", "square brackets"),
    ("C1CC", "ring isn't closed"),
    ("C(C)(C)(C)(C)C", "5 bonds"),
    ("c1cccc1", "aromatic ring"),
    ("C[Xx]", "isn't valid SMILES"),
])
def test_invalid_input_explains_why(text, expected):
    mol, problem = cu.check_smiles(text)
    assert mol is None and expected in problem


def test_valid_input_parses():
    mol, problem = cu.check_smiles("  " + ASPIRIN + "  ")
    assert mol is not None and problem is None


@pytest.mark.parametrize("smiles, expected", [
    ("CC(=O)[O-].[Na+]", "separate pieces"),
    ("C[Si](C)(C)C", "Si"),
    ("O", "no carbon"),
    ("[CH3]", "radical"),
    ("C[N+](C)(C)C", "net charge of +1"),
])
def test_input_warnings(smiles, expected):
    assert any(expected in w for w in cu.input_warnings(cu.parse_smiles(smiles)))


def test_size_warning():
    assert any("heavy atoms" in w for w in cu.input_warnings(cu.parse_smiles("C" * 200), max_heavy_atoms=164))


def test_ordinary_molecule_has_no_warnings():
    assert cu.input_warnings(cu.parse_smiles(ASPIRIN), max_heavy_atoms=164) == []


@pytest.mark.parametrize("smiles", ["C", "O", "[Fe+2]"])
def test_single_atom_heatmap_is_skipped(smiles):
    assert cu.logp_heatmap_png(cu.parse_smiles(smiles)) is None


def test_batch_reports_problem(bundle, catalogs, train_fps):
    table = cu.predict_batch(["C1CC"], bundle, catalogs, train_fps)
    assert "ring" in table.loc[0, "problem"]


# ------------------------------------------ parsing matches the training data
@pytest.mark.parametrize("plain, explicit", [
    ("CCO", "[H]OC([H])([H])C([H])([H])[H]"),
    (ASPIRIN, "[H]OC(=O)c1c([H])c([H])c([H])c([H])c1OC(=O)C([H])([H])[H]"),
])
def test_explicit_hydrogens_give_the_same_molecule(plain, explicit, bundle):
    """Input is parsed exactly as the training data was, however the hydrogens are written."""
    a, _ = cu.check_smiles(plain)
    b, _ = cu.check_smiles(explicit)
    assert cu.canonical_smiles(a) == cu.canonical_smiles(b)
    assert cu.featurize(a) == pytest.approx(cu.featurize(b))  # atom order differs, so tiny rounding differences
    assert cu.predict_logS(a, bundle) == pytest.approx(cu.predict_logS(b, bundle))


def test_check_smiles_matches_training_parser():
    for smiles in [ASPIRIN, "C[C@H](N)C(=O)O", "[2H]C", "CC(=O)[O-].[Na+]"]:
        mol, _ = cu.check_smiles(smiles)
        assert cu.canonical_smiles(mol) == cu.canonical_smiles(cu.parse_smiles(smiles))


# ----------------------------------------------------------------- size limits
def test_too_many_heavy_atoms_is_rejected():
    mol, problem = cu.check_smiles("C" * (cu.MAX_HEAVY_ATOMS + 1))
    assert mol is None and "heavy atoms" in problem


def test_too_long_smiles_is_rejected():
    mol, problem = cu.check_smiles("C" * (cu.MAX_SMILES_LENGTH + 1))
    assert mol is None and "characters long" in problem


def test_batch_stops_at_atom_budget(bundle, catalogs, train_fps, monkeypatch):
    monkeypatch.setattr(cu, "BATCH_ATOM_BUDGET", 20)  # aspirin has 13 heavy atoms
    table = cu.predict_batch([ASPIRIN, ASPIRIN, ASPIRIN], bundle, catalogs, train_fps)
    assert list(table["valid"]) == [True, False, False]
    assert "limit" in table.loc[1, "problem"]
