"""
app.py - Molecule Property Explorer (Streamlit front end).

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from chem_utils import (
    canonical_smiles,
    featurize,
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
    st.markdown(
        f"**Model:** random forest on ESOL (Delaney) descriptors  \n"
        f"**Test RMSE:** {bundle['test_rmse']:.2f} log units (random split)"
    )
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
    st.caption(
        f"Typical error is about ±{bundle['test_rmse']:.1f} log units, i.e. roughly a "
        f"{10 ** bundle['test_rmse']:.0f}x factor either way."
    )

    st.subheader("Descriptors used by the model")
    feats = featurize(mol)
    st.dataframe(
        pd.DataFrame({"Descriptor": feats.keys(), "Value": [round(v, 3) for v in feats.values()]}),
        hide_index=True,
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
