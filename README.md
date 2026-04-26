# Dataset Anonymization

Tool to anonymize and de-identify datasets: load a CSV, define metadata (identifiers, quasi-identifiers, sensitive attributes), and apply transformations such as random or order-preserving pseudonyms, numeric generalization, and perturbation. The original data is never modified; all changes apply to a working copy that you can export or reset.

**Requirements:** Python ≥3.12, [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

## Run

**CLI (placeholder):**

```bash
uv run python main.py
```

**Streamlit UI** (from project root):

```bash
uv run streamlit run src/interface/app.py
```

In the UI you can:

- **Load** a saved dataset (`.pkl`) from a folder (default `dataset_data`), or **create** a new one by uploading a CSV and filling metadata (column types, identifiers, quasi-identifiers, sensitive attributes). Validation must pass before continuing.
- **View** the first N rows (configurable) of the original or modified dataset.
- **Apply** de-identification (random or order-preserving pseudonyms, reverse OPE), numeric generalization or perturbation, masking, k-anonymity over quasi-identifiers, and reset the working copy.
- **Run differentially private queries** over the loaded dataset using an epsilon value and Laplace noise. Supported queries include row count, numeric sum, numeric mean, and histogram/count by category.
- **Export** the modified dataset to CSV (choose directory and filename) and **save** the manager to disk (`.pkl`) for later use.

## Project layout

| Path | Description |
|------|-------------|
| `src/dataset_anonymization/` | Core package: `DatasetManager`, `DatasetMetadata`, and anonymization logic. |
| `src/interface/` | Streamlit app that uses the manager (no business logic in the UI). |
| `docs/dataset-manager.md` | API and behaviour of the manager. |
