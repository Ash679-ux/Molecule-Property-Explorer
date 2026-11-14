"""
app.py - Molecule Property Explorer (Streamlit front end).

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from chem_utils import (
    DOMAIN_THRESHOLD,
    FEATURE_LABELS,
    band_of,
    bonds_between,
    build_alert_catalogs,
    canonical_smiles,
    featurize,
    fingerprint,
    has_p_equals_o,
    lipinski_report,
    load_model,
    logp_contributions,
    logp_heatmap_png,
    logS_to_mg_per_ml,
    nearest_neighbour,
    parse_smiles,
    predict_logS,
    qed_score,
    solubility_label,
    structural_alerts,
)

st.set_page_config(page_title="Molecule Property Explorer", page_icon="🧪", layout="wide")

EXAMPLES = {
    "Choose an example...": "",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O",
    "Triethyl phosphate (P=O)": "CCOP(=O)(OCC)OCC",
    "Erythromycin (breaks rule of five)": (
        "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)"
        "C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O"
    ),
    "Benzylidene rhodanine (PAINS alert)": "O=C1NC(=S)SC1=Cc1ccccc1",
}


@st.cache_resource
def get_model():
    """Load once and reuse across reruns (Streamlit reruns the script on every input)."""
    return load_model()


@st.cache_resource
def get_alert_catalogs():
    return build_alert_catalogs()


@st.cache_resource
def get_training_fingerprints():
    """Fingerprints of every training molecule, for the nearest-neighbour check."""
    return [fingerprint(parse_smiles(s)) for s in get_model()["train"]["smiles"]]


# ------------------------------------------------------------------ header
st.title("🧪 Molecule Property Explorer")
st.caption("Enter a molecule as SMILES to see its structure and predicted aqueous solubility.")

try:
    bundle = get_model()
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()

# ------------------------------------------------------------------- input
with st.sidebar:
    st.header("Input")
    example = st.selectbox("Examples", list(EXAMPLES.keys()))
    smiles = st.text_input("SMILES", value=EXAMPLES[example], placeholder="e.g. c1ccccc1")
    st.divider()
    cv = bundle["cv"]
    st.markdown(
        f"**Model:** gradient boosting on {len(bundle['features'])} RDKit descriptors, "
        f"trained on ESOL + AqSolDB ({bundle['n_train']:,} molecules)  \n"
        f"**Cross-validated RMSE on ESOL** (5 folds × 3 shuffles):  \n"
        f"- random folds: {cv['random']['RMSE']:.2f} log units  \n"
        f"- unseen ring frameworks: {cv['scaffold']['RMSE']:.2f} log units"
    )
    st.caption("Full analysis: [chem-ml-solubility](https://github.com/Ash679-ux/chem-ml-solubility)")
    # TODO (days 9-10): add the streamlit-ketcher drawing box here.

if not smiles:
    st.info("Pick an example or type a SMILES string in the sidebar to get started.")
    st.stop()

mol = parse_smiles(smiles)
if mol is None:
    st.error(f"Couldn't parse `{smiles}` as a molecule. Check the SMILES and try again.")
    st.stop()

# ----------------------------------------------------------------- results
col_struct, col_props = st.columns([1, 1])

with col_struct:
    st.subheader("Structure")
    st.image(Draw.MolToImage(mol, size=(400, 400)))
    st.code(canonical_smiles(mol), language=None)

with col_props:
    st.subheader("Predicted solubility")
    logS = predict_logS(mol, bundle)
    m1, m2 = st.columns(2)
    m1.metric("logS (mol/L)", f"{logS:.2f}")
    m2.metric("≈ mg/mL", f"{logS_to_mg_per_ml(logS, mol):.3g}")
    st.markdown(f"**{solubility_label(logS)}**")
    rmse_r, rmse_s = cv["random"]["RMSE"], cv["scaffold"]["RMSE"]
    st.caption(
        f"Typical error is about ±{rmse_r:.1f} log units (roughly a {10 ** rmse_r:.0f}x factor "
        f"either way), rising to ±{rmse_s:.1f} for ring frameworks the model hasn't seen."
    )
    if has_p_equals_o(mol):
        st.info(
            f"**Contains a P=O group (phosphate/phosphonate).** These were once this model's "
            f"worst class (off by about 2.5 log units). A P=O count and about 170 P=O compounds "
            f"from AqSolDB cut the error to about {cv['random']['MAE_PO']:.1f} log units, but "
            f"very soluble liquids such as triethyl phosphate can still be predicted too low."
        )

    # applicability domain: how close is this to anything the model was trained on?
    sim, nn_idx = nearest_neighbour(mol, get_training_fingerprints())
    train = bundle["train"]
    band = next(b for b in bundle["domain"] if (b["lo"], b["hi"]) == band_of(sim))
    if sim < DOMAIN_THRESHOLD:
        st.warning(
            f"**Unlike the training data** (nearest training molecule is only {sim:.2f} similar). "
            f"In cross-validation, molecules this dissimilar had an error of about "
            f"±{band['RMSE']:.1f} log units, so treat this prediction as a rough guess."
        )
    with st.expander(f"Nearest training molecule: {sim:.2f} similar"):
        nn_mol = parse_smiles(train["smiles"][nn_idx])
        st.image(Draw.MolToImage(nn_mol, size=(250, 250)))
        st.markdown(
            f"`{train['smiles'][nn_idx]}`  \n"
            f"Measured logS **{train['logS'][nn_idx]:.2f}** ({train['source'][nn_idx]})"
        )
        st.caption(
            "Similarity = Tanimoto similarity of Morgan fingerprints (radius 2), from 0 to 1. "
            f"Cross-validated error by similarity band: "
            + ", ".join(f"{b['lo']:.1f}–{b['hi']:.1f}: ±{b['RMSE']:.2f}" for b in bundle["domain"])
            + "."
        )

    st.subheader("Descriptors used by the model")
    feats = featurize(mol)
    importances = bundle["importances"]
    table = pd.DataFrame({
        "Descriptor": [FEATURE_LABELS[k] for k in feats],
        "Value": [round(v, 3) for v in feats.values()],
        "Importance": [importances[k] for k in feats],
    }).sort_values("Importance", ascending=False)
    st.dataframe(
        table,
        hide_index=True,
        height=320,
        column_config={
            "Importance": st.column_config.ProgressColumn(
                "Importance in model", format="%.2f", min_value=0, max_value=1
            )
        },
    )
    st.caption(
        "Importance = how much the error grows when that descriptor is shuffled, scaled to "
        "add up to 1. Scroll the table for all descriptors."
    )

# ------------------------------------------------------------- "why" view
st.divider()
st.header("Why this prediction?")
col_map, col_why = st.columns([1, 1])

with col_map:
    st.image(logp_heatmap_png(mol))

with col_why:
    weights = logp_contributions(mol)
    logp_share = importances["LogP"]
    st.metric("logP (Crippen)", f"{sum(weights):.2f}")
    st.markdown(
        f"logP is the model's most important descriptor (importance {logp_share:.2f}), so "
        f"the atoms driving logP drive most of the prediction.\n\n"
        f"- **Red** atoms raise logP: greasy parts that make the molecule **less** soluble\n"
        f"- **Blue** atoms lower logP: polar parts that make it **more** soluble"
    )
    greasy = sum(w for w in weights if w > 0)
    polar = sum(w for w in weights if w < 0)
    st.markdown(
        f"Greasy atoms add **{greasy:+.2f}**, polar atoms add **{polar:+.2f}**, "
        f"giving logP **{greasy + polar:.2f}**."
    )
    st.caption(
        "Contributions come from the Crippen logP method (Wildman & Crippen, 1999), with each "
        "hydrogen's share added to the atom it's bonded to. This explains the logP descriptor, "
        "not the whole model: the other descriptors still add their part."
    )

# ----------------------------------------------------------- drug-likeness
st.divider()
st.header("Drug-likeness")
col_qed, col_lip = st.columns([1, 2])

with col_qed:
    qed = qed_score(mol)
    st.metric("QED", f"{qed:.2f}")
    st.progress(qed)
    st.caption(
        "Quantitative Estimate of Drug-likeness (Bickerton et al., 2012): eight properties "
        "combined into one score from 0 to 1, where higher means more drug-like."
    )

with col_lip:
    lip = lipinski_report(mol)
    st.subheader("Lipinski's rule of five")
    st.dataframe(
        pd.DataFrame([
            {
                "Property": row["Property"],
                "Value": round(row["Value"], 2),
                "Limit": row["Limit"],
                "": "✅" if row["Passes"] else "❌",
            }
            for row in lip["rows"]
        ]),
        hide_index=True,
        width="stretch",
    )
    n = lip["violations"]
    verdict = f"{n} violation{'s' if n != 1 else ''}"
    if lip["drug_like"]:
        st.success(f"**{verdict}:** passes the rule of five (at most one violation allowed).")
    else:
        st.error(f"**{verdict}:** fails the rule of five, so poor oral absorption is more likely.")
    st.caption(
        "Uses Lipinski's original counts (NH + OH donors, N + O acceptors), so these can "
        "differ from the H-bond descriptors used by the solubility model. Many successful "
        "drugs break the rule, e.g. antibiotics and natural products; it's a guide, not a test."
    )

# ------------------------------------------------------- structural alerts
st.divider()
st.header("Structural alerts")
alerts = structural_alerts(mol, get_alert_catalogs())

if not alerts:
    st.success("No PAINS or Brenk alerts.")
else:
    n = len(alerts)
    st.warning(f"**{n} alert{'s' if n != 1 else ''}** matched. Highlighted atoms show where.")
    col_img, col_list = st.columns([1, 1])

    with col_list:
        options = ["All alerts"] + [f"{a['name']} ({a['source']})" for a in alerts]
        choice = st.selectbox("Highlight", options)
        st.dataframe(
            pd.DataFrame({
                "Alert": [a["name"] for a in alerts],
                "Source": [a["source"] for a in alerts],
                "Atoms matched": [len(a["atoms"]) for a in alerts],
            }),
            hide_index=True,
            width="stretch",
        )

    with col_img:
        chosen = alerts if choice == "All alerts" else [alerts[options.index(choice) - 1]]
        atoms = sorted({i for a in chosen for i in a["atoms"]})
        st.image(Draw.MolToImage(
            mol, size=(400, 400),
            highlightAtoms=atoms, highlightBonds=bonds_between(mol, atoms),
        ))

st.caption(
    "**PAINS** (Baell & Holloway, 2010) flags groups that often give false positives in "
    "screening assays. **Brenk** (Brenk et al., 2008) flags groups that are reactive, toxic "
    "or otherwise unwanted in drug candidates. These are flags for a chemist to look at, not "
    "verdicts: aspirin, a widely used drug, triggers the Brenk phenol ester alert."
)
