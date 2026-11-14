"""Molecule Property Explorer. Run with: streamlit run app.py"""

import pandas as pd
import streamlit as st
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher

import chem_utils as cu

st.set_page_config(page_title="Molecule Property Explorer", page_icon="🧪", layout="wide")

BATCH_LIMIT = 1000
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


# Loaded once per server, not on every rerun

@st.cache_resource
def get_model():
    return cu.load_model()


@st.cache_resource
def get_alert_catalogs():
    return cu.build_alert_catalogs()


@st.cache_resource
def get_training_fingerprints():
    return [cu.fingerprint(cu.parse_smiles(s)) for s in get_model()["train"]["smiles"]]


# Sidebar

def sidebar(bundle):
    """Returns (input mode, example name, typed SMILES or None)."""
    with st.sidebar:
        st.header("Input")
        example = st.selectbox("Examples", list(EXAMPLES))
        mode = st.radio("Enter molecules by", ["Typing SMILES", "Drawing it", "Uploading a CSV"])
        smiles = None
        if mode == "Typing SMILES":
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
    return mode, example, smiles


def drawing_box(example):
    st.subheader("Draw a molecule")
    # Keying on the example reloads the editor when a different example is picked.
    smiles = st_ketcher(EXAMPLES[example], height=450, key=f"ketcher-{example}")
    st.caption("Draw or edit the structure, then click **Apply** below the editor to update the results.")
    return smiles


# Batch mode

def batch_page(bundle):
    st.subheader("Batch predictions")
    st.markdown(
        f"Upload a CSV with a column called `smiles` (up to {BATCH_LIMIT:,} rows). "
        "You'll get one row of results per molecule, ready to download."
    )
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return
    try:
        data = pd.read_csv(upload)
    except Exception as err:
        st.error(f"Couldn't read that file as CSV: {err}")
        return

    column = next((c for c in data.columns if str(c).strip().lower() == "smiles"), None)
    if column is None:
        st.error(f"No `smiles` column found. Columns in the file: {', '.join(map(str, data.columns))}.")
        return
    if len(data) > BATCH_LIMIT:
        st.warning(f"The file has {len(data):,} rows; only the first {BATCH_LIMIT:,} are used.")

    with st.spinner("Predicting..."):
        results = cu.predict_batch(
            data[column].head(BATCH_LIMIT), bundle, get_alert_catalogs(), get_training_fingerprints()
        )
    invalid = int((~results["valid"]).sum())
    st.success(
        f"Done: {len(results) - invalid:,} molecules predicted"
        + (f", {invalid:,} invalid SMILES skipped." if invalid else ".")
    )
    st.dataframe(results, hide_index=True, width="stretch")
    st.download_button("Download results (CSV)", results.to_csv(index=False), "predictions.csv", "text/csv")


# Single-molecule sections

def solubility_section(mol, bundle):
    col_structure, col_results = st.columns(2)

    with col_structure:
        st.subheader("Structure")
        st.image(Draw.MolToImage(mol, size=(400, 400)))
        st.code(cu.canonical_smiles(mol), language=None)

    with col_results:
        st.subheader("Predicted solubility")
        logS = cu.predict_logS(mol, bundle)
        left, right = st.columns(2)
        left.metric("logS (mol/L)", f"{logS:.2f}")
        right.metric("≈ mg/mL", f"{cu.logS_to_mg_per_ml(logS, mol):.3g}")
        st.markdown(f"**{cu.solubility_label(logS)}**")

        cv = bundle["cv"]
        rmse_random, rmse_scaffold = cv["random"]["RMSE"], cv["scaffold"]["RMSE"]
        st.caption(
            f"Typical error is about ±{rmse_random:.1f} log units (roughly a {10 ** rmse_random:.0f}x "
            f"factor either way), rising to ±{rmse_scaffold:.1f} for ring frameworks the model hasn't seen."
        )
        if cu.has_phosphoryl(mol):
            st.info(
                "**Contains a P=O group (phosphate/phosphonate).** These were once this model's worst "
                "class (off by about 2.5 log units). A P=O count and about 170 P=O compounds from "
                f"AqSolDB cut the error to about {cv['random']['MAE_PO']:.1f} log units, but very "
                "soluble liquids such as triethyl phosphate can still be predicted too low."
            )
        domain_check(mol, bundle)
        descriptor_table(mol, bundle)


def domain_check(mol, bundle):
    similarity, idx = cu.nearest_neighbour(mol, get_training_fingerprints())
    band = next(b for b in bundle["domain"] if (b["lo"], b["hi"]) == cu.band_of(similarity))
    if similarity < cu.DOMAIN_THRESHOLD:
        st.warning(
            f"**Unlike the training data** (nearest training molecule is only {similarity:.2f} similar). "
            f"In cross-validation, molecules this dissimilar had an error of about "
            f"±{band['RMSE']:.1f} log units, so treat this prediction as a rough guess."
        )

    train = bundle["train"]
    with st.expander(f"Nearest training molecule: {similarity:.2f} similar"):
        st.image(Draw.MolToImage(cu.parse_smiles(train["smiles"][idx]), size=(250, 250)))
        st.markdown(
            f"`{train['smiles'][idx]}`  \n"
            f"Measured logS **{train['logS'][idx]:.2f}** ({train['source'][idx]})"
        )
        bands = ", ".join(f"{b['lo']:.1f}–{b['hi']:.1f}: ±{b['RMSE']:.2f}" for b in bundle["domain"])
        st.caption(
            "Similarity is the Tanimoto similarity of Morgan fingerprints (radius 2), from 0 to 1. "
            f"Cross-validated error by similarity: {bands}."
        )


def descriptor_table(mol, bundle):
    st.subheader("Descriptors used by the model")
    features = cu.featurize(mol)
    table = pd.DataFrame({
        "Descriptor": [cu.FEATURE_LABELS[name] for name in features],
        "Value": [round(value, 3) for value in features.values()],
        "Importance": [bundle["importances"][name] for name in features],
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
        "Importance is how much the error grows when that descriptor is shuffled, scaled to add up to 1."
    )


def why_section(mol, bundle):
    st.divider()
    st.header("Why this prediction?")
    col_map, col_text = st.columns(2)

    with col_map:
        st.image(cu.logp_heatmap_png(mol))

    with col_text:
        weights = cu.logp_contributions(mol)
        greasy = sum(w for w in weights if w > 0)
        polar = sum(w for w in weights if w < 0)
        st.metric("logP (Crippen)", f"{greasy + polar:.2f}")
        st.markdown(
            f"logP is the model's most important descriptor (importance "
            f"{bundle['importances']['LogP']:.2f}), so the atoms driving logP drive most of the "
            "prediction.\n\n"
            "- **Red** atoms raise logP: greasy parts that make the molecule **less** soluble\n"
            "- **Blue** atoms lower logP: polar parts that make it **more** soluble\n\n"
            f"Greasy atoms add **{greasy:+.2f}**, polar atoms add **{polar:+.2f}**."
        )
        st.caption(
            "Contributions come from the Crippen logP method (Wildman & Crippen, 1999). This explains "
            "the logP descriptor, not the whole model: the other descriptors still add their part."
        )


def drug_likeness_section(mol):
    st.divider()
    st.header("Drug-likeness")
    col_qed, col_lipinski = st.columns([1, 2])

    with col_qed:
        qed = cu.qed_score(mol)
        st.metric("QED", f"{qed:.2f}")
        st.progress(qed)
        st.caption(
            "Quantitative Estimate of Drug-likeness (Bickerton et al., 2012): eight properties "
            "combined into one score from 0 to 1, where higher means more drug-like."
        )

    with col_lipinski:
        report = cu.rule_of_five(mol)
        st.subheader("Lipinski's rule of five")
        st.dataframe(
            pd.DataFrame([
                {
                    "Property": row["Property"],
                    "Value": round(row["Value"], 2),
                    "Limit": row["Limit"],
                    "": "✅" if row["Passes"] else "❌",
                }
                for row in report["rows"]
            ]),
            hide_index=True,
            width="stretch",
        )
        n = report["violations"]
        verdict = f"{n} violation{'s' if n != 1 else ''}"
        if report["drug_like"]:
            st.success(f"**{verdict}:** passes the rule of five (at most one violation allowed).")
        else:
            st.error(f"**{verdict}:** fails the rule of five, so poor oral absorption is more likely.")
        st.caption(
            "Uses Lipinski's original counts (NH + OH donors, N + O acceptors), so these can differ "
            "from the model's H-bond descriptors. Many successful drugs break the rule, such as "
            "antibiotics and natural products: it's a guide, not a test."
        )


def alerts_section(mol):
    st.divider()
    st.header("Structural alerts")
    alerts = cu.structural_alerts(mol, get_alert_catalogs())

    if not alerts:
        st.success("No PAINS or Brenk alerts.")
    else:
        n = len(alerts)
        st.warning(f"**{n} alert{'s' if n != 1 else ''}** matched. Highlighted atoms show where.")
        col_image, col_list = st.columns(2)

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

        with col_image:
            shown = alerts if choice == "All alerts" else [alerts[options.index(choice) - 1]]
            atoms = sorted({i for alert in shown for i in alert["atoms"]})
            st.image(Draw.MolToImage(
                mol, size=(400, 400), highlightAtoms=atoms, highlightBonds=cu.bonds_between(mol, atoms)
            ))

    st.caption(
        "**PAINS** (Baell & Holloway, 2010) flags groups that often give false positives in screening "
        "assays. **Brenk** (Brenk et al., 2008) flags groups that are reactive, toxic or otherwise "
        "unwanted in drug candidates. These are flags for a chemist to look at, not verdicts: aspirin, "
        "a widely used drug, triggers the Brenk phenol ester alert."
    )


def main():
    st.title("🧪 Molecule Property Explorer")
    st.caption(
        "Type or draw a molecule, or upload a CSV, to see predicted solubility, "
        "drug-likeness and structural alerts."
    )
    try:
        bundle = get_model()
    except (FileNotFoundError, ValueError) as err:
        st.error(str(err))
        return

    mode, example, smiles = sidebar(bundle)
    if mode == "Uploading a CSV":
        batch_page(bundle)
        return
    if mode == "Drawing it":
        smiles = drawing_box(example)

    if not smiles:
        st.info("Pick an example, type a SMILES string or draw a molecule to get started.")
        return
    mol = cu.parse_smiles(smiles)
    if mol is None:
        st.error(f"Couldn't parse `{smiles}` as a molecule. Check the SMILES and try again.")
        return

    solubility_section(mol, bundle)
    why_section(mol, bundle)
    drug_likeness_section(mol)
    alerts_section(mol)


main()
