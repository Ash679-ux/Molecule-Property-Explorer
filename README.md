# Molecule Property Explorer

A Streamlit web app that predicts and explains molecular properties from a SMILES string:
structure, predicted aqueous solubility (ESOL model), drug-likeness and structural alerts.

Builds on my earlier analysis project, [chem-ml-solubility](https://github.com/Ash679-ux/chem-ml-solubility).

> 🚧 Work in progress. Full write-up, screenshots and the live demo link will go here.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py      # trains the model and saves it to models/esol_model.joblib
python chem_utils.py       # smoke test
streamlit run app.py
```

## Data

`data/delaney-processed.csv`: the ESOL dataset (Delaney, 2004) as distributed by
[DeepChem / MoleculeNet](https://github.com/deepchem/deepchem).
