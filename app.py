"""
app.py - Molecule Property Explorer (Streamlit front end).

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from chem_utils import (
    FEATURE_LABELS,
    canonical_smiles,
    featurize,
    has_p_equals_o,
    load_model,
    logS_to_mg_per_ml,
    parse_smiles,
    predict_logS,
    solubility_label,
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
}


@st.cache_resource
def get_model():
    """Load once and reuse across reruns (Streamlit reruns the script on every input)."""
    return load_model()


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
        f"**Model:** random forest on 7 RDKit descriptors, trained on ESOL (1,128 molecules)  \n"
        f"**Cross-validated RMSE** (5 folds × 3 shuffles):  \n"
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
        st.warning(
            "**Contains a P=O group (phosphate/phosphonate).** In cross-validation this model "
            "was off by about 2.6 log units on these compounds, and predicted them as less "
            "soluble than measured. Treat this prediction with caution."
        )

    st.subheader("Descriptors used by the model")
    feats = featurize(mol)
    importances = bundle["importances"]
    st.dataframe(
        pd.DataFrame({
            "Descriptor": [FEATURE_LABELS[k] for k in feats],
            "Value": [round(v, 3) for v in feats.values()],
            "Importance": [importances[k] for k in feats],
        }),
        hide_index=True,
        column_config={
            "Importance": st.column_config.ProgressColumn(
                "Importance in model", format="%.2f", min_value=0, max_value=1
            )
        },
    )

# -------------------------------------------------------- coming next
# TODO (days 5-6): drug-likeness panel - Lipinski rule of five + QED
#                  (from rdkit.Chem import QED, Lipinski)
# TODO (days 5-6): structural alerts - PAINS/Brenk via
#                  rdkit.Chem.FilterCatalog, highlight matched atoms
# TODO (days 7-8): "why" view - Crippen atom contributions drawn with
#                  rdkit.Chem.Draw.SimilarityMaps
# TODO (extra):    applicability-domain warning using Tanimoto similarity
#                  to the nearest training molecule
