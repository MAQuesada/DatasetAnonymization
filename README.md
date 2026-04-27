# Dataset Anonymization

Tool to anonymize and de-identify datasets: load a CSV, define metadata (identifiers, quasi-identifiers, sensitive attributes), and apply transformations such as random or order-preserving pseudonyms, numeric generalization, and perturbation. The original data is never modified; all changes apply to a working copy that you can export or reset.

**Requirements:** Python ≥3.12, [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

## Run Dataset Anonymization

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
- **Apply** de-identification (random or order-preserving pseudonyms, reverse OPE), numeric generalization or perturbation, and reset the working copy.
- **Export** the modified dataset to CSV (choose directory and filename) and **save** the manager to disk (`.pkl`) for later use.

## Run Dataset Creation

**Streamlit UI** (from project root):

```bash
uv run streamlit run src/interface/ui_creation/app_dataset_creation.py --server.port 8502
```

We used a different port to avoid some conflicts with the Dataset Anonymization execution.

In the UI you can:

- **Select the required features for your fake dataset**. This includes selecting the number of rows, the desired columns and the random data appearance percentage. You can select the following columns: **Name**, **First surname**, **Second Surname**, **Age**, **Gender** and **Municipality of Malaga**.

- **Create custom columns**. You can dynamically add new columns to the dataset by choosing between two types:
  
  - **Categorical columns**:  
    Define a column name, a list of possible values (comma-separated), and optionally a list of weights (also comma-separated) to control how frequently each value appears. If weights are not provided or are invalid, all values will be generated with equal probability.
  
  - **Numerical columns**:  
    Define a column name, a minimum and maximum value, and select a distribution type (`uniform`, `normal`, `binomial`, or `lognormal`). The values will be generated according to the selected distribution within the specified range.

  You can add multiple custom columns and also remove them before generating the dataset.

- **Generate** the fake dataset. Depending on whether it is the first execution or if the number of samples is quite large, it msy take some time

When the dataset is generated, you will see:

- A **view** of the generated dataset. You can navigate through 10 samples per page. 
- A **Go back** button, which takes you back to the previous UI so you can generate a new dataset again.
- A **Download** button to download the dataframe as a .csv. 

It is recommended to **generate one sample the first time you run it**, and then go back and generate the desired dataset. This is because **the first execution takes some time**. 

## Project layout

| Path | Description |
|------|-------------|
| `src/dataset_anonymization/` | Core package: `DatasetManager`, `DatasetMetadata`, and anonymization logic. |
| `src/dataset_creation/` | Core package: `generate_dataset` function. |
| `src/interface/` | Streamlit app that uses the manager (no business logic in the UI). It also contains Streamlit app that uses the creator |
| `docs/dataset-manager.md` | API and behaviour of the manager. |
| `docs/dataset-creation.md` | Behaviour of the generated dataset. |
