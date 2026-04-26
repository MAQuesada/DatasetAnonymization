# DatasetManager — Structure and Functionality

The `DatasetManager` is the domain-layer component that holds the dataset and applies all anonymization and de-identification operations. It does not depend on the UI (CLI, web, or API); only the manager’s methods are called from the interface.

---

## Structure

### Classes and exception

| Name | Description |
|------|-------------|
| **`DatasetMetadata`** | Dataclass that describes the dataset: column names and types, and which columns are identifiers, quasi-identifiers, or sensitive. |
| **`DatasetManager`** | Main class. Keeps the original dataset and a working copy; all transformations change only the working copy. |
| **`DatasetManagerLoadError`** | Exception raised when loading a saved manager fails (missing file, invalid file, or validation error). |

### Internal state

- **Original dataset** (`pandas.DataFrame`): Loaded once and never modified.
- **Working dataset** (`pandas.DataFrame`): Copy used for all transformations; can be reset to match the original.
- **Metadata** (`DatasetMetadata`): Column types and role lists (identifiers, quasi-identifiers, sensitive attributes).
- **OPE mappings** (internal): Stores reversible mappings for order-preserving pseudonyms; cleared on reset.

On construction, the manager checks that every column referenced in the metadata exists in the dataset. If any is missing, it raises `ValueError`.

---

## DatasetMetadata

Used to create a `DatasetManager`. All listed columns must exist in the DataFrame.

| Field | Description |
|-------|-------------|
| `column_types` | Dict mapping each column name to `"numeric"` or `"categorical"`. |
| `identifiers` | Column names that uniquely identify a person (e.g. ID, SSN). |
| `quasi_identifiers` | Column names that can help re-identify when combined (e.g. age, zip). |
| `sensitive_attributes` | Column names to protect (e.g. salary, diagnosis). |

---

## Public API

### Dataset access and reset

| Method | Description |
|--------|-------------|
| `get_original_dataset()` | Returns a **copy** of the original dataset (read-only use). |
| `get_working_dataset()` | Returns a **copy** of the current working dataset. |
| `reset_working_dataset()` | Restores the working dataset from the original and clears OPE mappings. |
| `export_working_to_csv(path, **kwargs)` | Writes the current working dataset to a CSV file at `path`. Creates parent directories if needed. Extra arguments are passed to `pandas.DataFrame.to_csv` (e.g. `sep`, `index`, `encoding`). |

### Persistence

| Method | Description |
|--------|-------------|
| `save(path)` | Saves the full manager (original, working, metadata, OPE mappings) to a binary file at `path`. Creates parent directories if needed. |
| `DatasetManager.load(path)` | Class method. Loads a manager from a file, runs validation, and returns the instance. Raises `DatasetManagerLoadError` if the file is missing, invalid, or metadata/dataset validation fails. |

### De-identification

| Method | Description |
|--------|-------------|
| `deidentify_with_random_pseudonyms(identifier_column=None)` | Replaces identifier values with random, non-reversible pseudonyms (e.g. `ID-<random>`). `identifier_column`: one column name from metadata, or `None` to apply to all identifiers. |
| `deidentify_with_order_preserving_pseudonyms(identifier_column=None)` | Replaces identifier values with reversible, order-preserving pseudonyms (rank-based). Mappings are stored for reversal. Same `identifier_column` semantics. Not cryptographically secure; for real use a proper OPE scheme should be used. |
| `reverse_order_preserving_pseudonyms(identifier_column=None)` | Restores original values for columns previously pseudonymized with `deidentify_with_order_preserving_pseudonyms`. Uses the stored mapping; after reversal the mapping for that column is removed. `identifier_column`: one column name or `None` to reverse all columns that have a mapping. Raises if the column has no stored mapping. |

### Anonymization (numeric columns)

| Method | Description |
|--------|-------------|
| `generalize_numeric_column(column, bins=10, include_lowest=True)` | Bins the column and replaces values with the bin midpoint. Integer columns remain integer (rounded); float columns remain float. Column must be declared `"numeric"` in metadata. |
| `perturb_numeric_column(column, noise_std=0.05, random_state=None)` | Adds Gaussian noise scaled to the column range (max − min). `noise_std` is a fraction of that range (e.g. `0.05` → 5%). Values are clipped to the column min/max; int columns stay int (rounded), float stay float. |
| `mask_column(column, strategy, k=None, regex_pattern=None)` | Masks one column by treating each value as string. Strategies: first K chars, last K chars, before first space, after last space, or regex-based replacement with `*`. |

### Utility / metrics

| Method | Description |
|--------|-------------|
| `get_column_statistics(use_working=True)` | Returns a DataFrame with one row per column and main statistics. All columns: count, unique_count, nulls. Numeric: plus mean, min, max, std, median. Non-applicable cells are NaN. Use `use_working=False` for the original dataset. |
| `compute_precision_privacy_tradeoff()` | Returns `(precision_pct, privacy_pct)` in [0, 100]. Current implementation is a placeholder: precision is based on unchanged cells (sensitive columns weighted more), privacy = 100 − precision. This metric is intended to be replaced by a proper utility/privacy definition. |

---

## Validation rules

- At construction and after load: every column in `column_types`, `identifiers`, `quasi_identifiers`, and `sensitive_attributes` must exist in the dataset.
- Numeric operations (`generalize_numeric_column`, `perturb_numeric_column`) require the column to exist and to be declared `"numeric"` in metadata.
- De-identification methods require the given column (or all identifiers) to be in `metadata.identifiers`.

Violations raise `ValueError`, `KeyError`, `TypeError`, or `DatasetManagerLoadError` as appropriate.
