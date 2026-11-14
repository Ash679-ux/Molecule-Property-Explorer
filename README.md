# Molecule Property Explorer

A Streamlit web app that predicts and explains molecular properties from a SMILES string:
structure, predicted aqueous solubility and drug-likeness.

Builds on my earlier analysis project, [chem-ml-solubility](https://github.com/Ash679-ux/chem-ml-solubility).

> 🚧 Work in progress. Screenshots and the live demo link will go here.

## Features

- **Predicted solubility** (logS and mg/mL) with a plain-English label and the model's typical error
- **Descriptor table** showing the 25 descriptors the model uses, sorted by importance
- **Applicability-domain warning** when a molecule is unlike anything in the training data, with the nearest training molecule and its measured logS
- **P=O note** for phosphates and phosphonates, the model's historically hardest class
- **Drug-likeness:** QED score and Lipinski's rule of five
- **Structural alerts:** PAINS and Brenk substructure flags, with the matched atoms highlighted
- **Why this prediction?** A heatmap of each atom's contribution to logP, the model's most important descriptor

## Model

Histogram gradient boosting on 25 RDKit descriptors (the 7 from chem-ml-solubility, a P=O count,
ring counts, sp³ fraction, molar refractivity, surface area, complexity and element counts),
trained on ESOL + AqSolDB (8,794 molecules).

Scored on ESOL with the same 5-fold cross-validation (3 shuffles) as chem-ml-solubility. AqSolDB
is only ever added to the training folds, with every ESOL compound removed, and in scaffold folds
every compound sharing a test-fold ring framework removed too.

| Model | RMSE, random folds | RMSE, unseen ring frameworks | Error on P=O compounds |
| --- | --- | --- | --- |
| Random forest, 7 descriptors, ESOL only (chem-ml-solubility) | 0.709 | 0.804 | 2.49 |
| This app: gradient boosting, 25 descriptors, ESOL + AqSolDB | 0.61 | 0.69 | 0.5 |

Errors are in log units of mol/L. Adding AqSolDB fixed the P=O compounds; the extra descriptors and
gradient boosting improved everything else. Ulrich et al.'s dataset scored the same overall but worse
on phosphates, because its logS > 0.5 cutoff removes the most soluble ones.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py      # downloads AqSolDB, trains (about 1 minute), saves models/esol_model.joblib
python chem_utils.py       # smoke test
streamlit run app.py
```

On WSL, training caps itself at 4 threads, because using every core can stall gradient boosting.
Override with `OMP_NUM_THREADS=8 python train_model.py`.

## Data

- `data/delaney-processed.csv`: ESOL (Delaney, 2004) as distributed by
  [DeepChem / MoleculeNet](https://github.com/deepchem/deepchem)
- AqSolDB (Sorkun et al., 2019), downloaded on first run from
  [github.com/mcsorkun/AqSolDB](https://github.com/mcsorkun/AqSolDB)
