# Molecular Hydration Free Energy Prediction Using Machine Learning

A machine learning study for predicting molecular hydration free energy from simple molecular descriptors calculated using RDKit.

This project compares multiple machine learning model families, including linear models, kernel-based methods, ensemble tree models, and a neural network.

---

## Project Overview

Hydration free energy is an important molecular property in computational chemistry and molecular simulation.

In this project, molecular structures represented by SMILES strings are converted into numerical molecular descriptors using RDKit. These descriptors are then used as inputs to several machine learning models.

The main goal is to compare different machine learning approaches using the same molecular representation and dataset.

---

## Dataset

The project uses the **FreeSolv** dataset.

- Number of molecules: **642**
- Molecular representation: **SMILES**
- Target variable: **y**
- Missing values: **None**
- Duplicate rows: **None**
- Invalid SMILES: **None**

### Molecular Descriptors

Four basic molecular descriptors were calculated using RDKit:

| Descriptor | Description |
|---|---|
| MolWt | Molecular weight |
| LogP | Octanol/water partition coefficient |
| HBD | Number of hydrogen-bond donors |
| HBA | Number of hydrogen-bond acceptors |

---

## Machine Learning Workflow

```text
SMILES
   ↓
RDKit Molecular Objects
   ↓
Molecular Descriptors
   ↓
Train / Test Split
   ↓
Feature Scaling where required
   ↓
Multiple ML Models
   ↓
Prediction
   ↓
MAE / RMSE / R²
   ↓
Model Comparison
