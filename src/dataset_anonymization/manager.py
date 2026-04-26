from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


class DatasetManagerLoadError(Exception):
    """Raised when a DatasetManager cannot be loaded from disk (file or validation failure)."""


@dataclass
class DatasetMetadata:
    """Container for dataset metadata used by DatasetManager."""

    column_types: Dict[str, str]
    identifiers: List[str] = field(default_factory=list)
    quasi_identifiers: List[str] = field(default_factory=list)
    sensitive_attributes: List[str] = field(default_factory=list)


class DatasetManager:
    """Domain-layer manager responsible for dataset state and anonymization operations."""

    def __init__(self, original_df: pd.DataFrame, metadata: DatasetMetadata) -> None:
        """Initialize the manager with an original dataset and associated metadata.

        The original dataset is never modified. All transformations are applied
        to the internal working copy.
        """
        self._original_df = original_df.copy(deep=True)
        self._working_df = original_df.copy(deep=True)
        self._metadata = metadata
        self._validate_metadata_consistency()

        # Stores reversible mappings for pseudonyms applied with order-preserving "encryption".
        self._ope_mappings: Dict[str, Dict[Any, Any]] = {}

    @property
    def metadata(self) -> DatasetMetadata:
        """Return the dataset metadata."""
        return self._metadata

    @property
    def ope_mapping_columns(self) -> List[str]:
        """Return column names that have a stored OPE mapping (reversible)."""
        return list(self._ope_mappings.keys())

    def get_original_dataset(self) -> pd.DataFrame:
        """Return a copy of the original dataset, which must not be modified."""
        return self._original_df.copy(deep=True)

    def get_working_dataset(self) -> pd.DataFrame:
        """Return a copy of the current working dataset."""
        return self._working_df.copy(deep=True)

    def reset_working_dataset(self) -> None:
        """Reset the working dataset so it matches the original dataset."""
        self._working_df = self._original_df.copy(deep=True)
        self._ope_mappings.clear()

    def export_working_to_csv(self, path: str | Path, **kwargs: Any) -> None:
        """Write the current working dataset (with all transformations) to a CSV file.

        Parameters
        ----------
        path:
            File path where the CSV will be written. Parent directories are created if needed.
        **kwargs:
            Optional arguments passed through to pandas.DataFrame.to_csv (e.g. sep, index, encoding).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._working_df.to_csv(path, **kwargs)

    def save(self, path: str | Path) -> None:
        """Persist this manager to disk as a binary file (RNF-DM-04).

        Parameters
        ----------
        path:
            File path where the manager will be saved. Existing file is overwritten.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> DatasetManager:
        """Load a DatasetManager from a binary file (RNF-DM-04).

        Parameters
        ----------
        path:
            File path from which to load the manager.

        Returns
        -------
        DatasetManager:
            Restored instance.

        Raises
        ------
        DatasetManagerLoadError:
            If the file cannot be read, the content is not a DatasetManager,
            or metadata/dataset validation fails after load.
        """
        path = Path(path)
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except FileNotFoundError as e:
            raise DatasetManagerLoadError(f"Cannot load DatasetManager: file not found: {path}") from e
        except pickle.UnpicklingError as e:
            raise DatasetManagerLoadError(f"Cannot load DatasetManager: invalid or corrupted file: {path}") from e

        if not isinstance(obj, cls):
            raise DatasetManagerLoadError(
                f"Cannot load DatasetManager: file contains {type(obj).__name__}, not DatasetManager."
            )

        try:
            obj._validate_metadata_consistency()
        except ValueError as e:
            raise DatasetManagerLoadError(
                f"Cannot load DatasetManager: validation failed (metadata columns must exist in dataset): {e}"
            ) from e

        return obj

    def deidentify_with_random_pseudonyms(
        self,
        identifier_column: Optional[str] = None,
    ) -> None:
        """Replace identifiers with random, non-reversible pseudonyms (one-way encryption).

        Identifier columns may be string or numeric; values are replaced in place
        with pseudonyms of the form ID-<random>. Output is always string.

        Parameters
        ----------
        identifier_column:
            Name of the identifier column in metadata.identifiers.
            If None, all identifier columns are processed.
        """
        target_identifiers = self._select_identifiers(identifier_column)

        rng = np.random.default_rng()
        for column in target_identifiers:
            unique_values = self._working_df[column].unique()
            pseudonyms = {
                original: f"ID-{token}"
                for original, token in zip(
                    unique_values,
                    rng.integers(low=0, high=2**63 - 1, size=len(unique_values), dtype="int64"),
                )
            }
            self._working_df[column] = self._working_df[column].replace(pseudonyms)

    def deidentify_with_order_preserving_pseudonyms(
        self,
        identifier_column: Optional[str] = None,
    ) -> None:
        """Replace identifiers with reversible, order-preserving pseudonyms (OPE-style).

        Identifier columns may be string or numeric. Values are sorted (strings
        lexicographically, numbers numerically) and replaced by integer ranks; the
        working column becomes numeric until reversed. This implementation is not
        cryptographically secure; use a proper OPE scheme for production.

        Parameters
        ----------
        identifier_column:
            Name of the identifier column in metadata.identifiers.
            If None, all identifier columns are processed.
        """
        target_identifiers = self._select_identifiers(identifier_column)

        for column in target_identifiers:
            series = self._working_df[column]
            uniques = series.dropna().unique()
            unique_sorted = sorted(uniques, key=lambda x: (str(type(x).__name__), x))
            mapping: Dict[Any, Any] = {}
            for rank, value in enumerate(unique_sorted, start=1):
                mapping[value] = rank

            # Preserve missing values without mapping.
            mapped = series.replace(mapping)
            self._working_df[column] = mapped.where(~series.isna(), other=series)
            self._ope_mappings[column] = mapping

    def reverse_order_preserving_pseudonyms(
        self,
        identifier_column: Optional[str] = None,
    ) -> None:
        """Restore original identifier values from order-preserving pseudonyms.

        Only columns previously processed with deidentify_with_order_preserving_pseudonyms
        have a stored mapping; others are skipped. Same identifier_column semantics:
        None means all identifier columns that have a mapping, otherwise the given column.

        Parameters
        ----------
        identifier_column:
            Name of the identifier column to reverse, or None for all that have a mapping.
        """
        if identifier_column is None:
            columns_to_reverse = list(self._ope_mappings.keys())
        else:
            if identifier_column not in self._metadata.identifiers:
                raise ValueError(
                    f"Column '{identifier_column}' is not in metadata.identifiers."
                )
            if identifier_column not in self._ope_mappings:
                raise ValueError(
                    f"No reversible mapping for column '{identifier_column}'. "
                    "Apply deidentify_with_order_preserving_pseudonyms first."
                )
            columns_to_reverse = [identifier_column]

        for column in columns_to_reverse:
            mapping = self._ope_mappings.get(column)
            if mapping is None:
                continue
            inverse = {v: k for k, v in mapping.items()}
            series = self._working_df[column]
            restored = series.replace(inverse)
            self._working_df[column] = restored.where(~series.isna(), other=series)
            del self._ope_mappings[column]

    def generalize_numeric_column(
        self,
        column: str,
        bins: int = 10,
        include_lowest: bool = True,
    ) -> None:
        """Generalize a numeric column by replacing values with the midpoint of their bin.

        Each value is assigned to a bin; the value is replaced by the midpoint of that
        interval (e.g. bin (27.7, 33.4] -> 30.55). Original type is preserved: integer
        columns get rounded midpoints as integers, float columns stay float.

        Parameters
        ----------
        column:
            Name of the numeric column in the working dataset.
        bins:
            Number of intervals for binning. Defaults to 10.
        include_lowest:
            Whether to include the lowest value in the first interval.
        """
        self._ensure_column_exists(column)
        self._ensure_numeric_column(column)

        series = self._working_df[column]
        binned = pd.cut(series.astype(float), bins=bins, include_lowest=include_lowest)
        binned_series = pd.Series(binned, index=series.index)
        midpoints = binned_series.apply(lambda iv: float("nan") if pd.isna(iv) else iv.mid)
        is_int = pd.api.types.is_integer_dtype(series.dtype)

        if is_int:
            generalized = midpoints.round().astype(series.dtype)
        else:
            generalized = midpoints
        self._working_df[column] = generalized.where(series.notna(), other=series)

    def perturb_numeric_column(
        self,
        column: str,
        noise_std: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        """Perturb a numeric column by adding scale-aware random noise.

        Noise is applied relative to the column range (max - min), so the same
        parameter works for different scales (e.g. grades 0-100 vs salary
        20_000-60_000). Values are clipped to the column min/max and the
        column type is preserved (int stays int after rounding, float stays float).

        Parameters
        ----------
        column:
            Name of the numeric column in the working dataset.
        noise_std:
            Standard deviation of the Gaussian noise as a fraction of the
            column range (max - min). E.g. 0.05 means noise has std = 5% of
            the range; 0.1 means 10%. Default 0.05.
        random_state:
            Optional random seed for reproducibility.
        """
        self._ensure_column_exists(column)
        self._ensure_numeric_column(column)

        series = self._working_df[column]
        col_min = series.min()
        col_max = series.max()
        col_range = col_max - col_min
        if col_range == 0:
            col_range = 1.0
        effective_std = float(noise_std * col_range)
        is_int = pd.api.types.is_integer_dtype(series.dtype)

        rng = np.random.default_rng(random_state)
        noise = rng.normal(loc=0.0, scale=effective_std, size=len(self._working_df))
        perturbed = series.astype(float) + noise
        perturbed = perturbed.clip(lower=col_min, upper=col_max)

        if is_int:
            self._working_df[column] = perturbed.round().astype(series.dtype)
        else:
            self._working_df[column] = perturbed

    def mask_column(
        self,
        column: str,
        strategy: str,
        k: Optional[int] = None,
        regex_pattern: Optional[str] = None,
    ) -> None:
        """Mask one column by treating each non-null value as a string.

        Supported strategies:
        - mask_first_k
        - mask_last_k
        - mask_before_first_space
        - mask_after_last_space
        - mask_regex

        Parameters
        ----------
        column:
            Target column name in the working dataset.
        strategy:
            One masking strategy from the supported list.
        k:
            Number of characters to mask for first/last-k strategies.
        regex_pattern:
            Regular expression to replace with '*' for mask_regex strategy.
        """
        self._ensure_column_exists(column)

        valid_strategies = {
            "mask_first_k",
            "mask_last_k",
            "mask_before_first_space",
            "mask_after_last_space",
            "mask_regex",
        }
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown masking strategy '{strategy}'.")

        if strategy in {"mask_first_k", "mask_last_k"}:
            if k is None or k < 0:
                raise ValueError("Parameter 'k' must be a non-negative integer.")

        if strategy == "mask_regex":
            if not regex_pattern:
                raise ValueError("Parameter 'regex_pattern' is required for mask_regex strategy.")
            try:
                compiled_pattern = re.compile(regex_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e
        else:
            compiled_pattern = None

        def _mask_value(value: Any) -> Any:
            if pd.isna(value):
                return value

            text = str(value)

            if strategy == "mask_first_k":
                n = min(k or 0, len(text))
                return ("*" * n) + text[n:]

            if strategy == "mask_last_k":
                n = min(k or 0, len(text))
                if n == 0:
                    return text
                return text[:-n] + ("*" * n)

            if strategy == "mask_before_first_space":
                first_space = text.find(" ")
                if first_space == -1:
                    return text
                return ("*" * first_space) + text[first_space:]

            if strategy == "mask_after_last_space":
                last_space = text.rfind(" ")
                if last_space == -1:
                    return text
                return text[: last_space + 1] + ("*" * (len(text) - last_space - 1))

            # strategy == "mask_regex"
            return compiled_pattern.sub("*", text)  # type: ignore[union-attr]

        self._working_df[column] = self._working_df[column].apply(_mask_value)

    def compute_precision_privacy_tradeoff(self) -> Tuple[float, float]:
        """Compute a toy precision vs. privacy tradeoff metric.

        Returns
        -------
        precision_pct, privacy_pct:
            A pair of percentages in the range [0, 100].

        NOTE
        ----
        #######################################################################
        # TEMPORARY METRIC IMPLEMENTATION                                     #
        # This is a simple placeholder that MUST be replaced with a proper   #
        # utility/privacy evaluation strategy aligned with the project needs.#
        # Current behavior:                                                  #
        #   - Precision is the percentage of unchanged cells between         #
        #     original and working datasets.                                 #
        #   - Cells in sensitive attributes are weighted twice to reflect    #
        #     their higher privacy impact.                                   #
        #   - Privacy is computed as (100 - precision).                      #
        #######################################################################
        """
        original = self._original_df
        working = self._working_df

        if not original.columns.equals(working.columns) or len(original) != len(working):
            raise ValueError("Original and working datasets are not directly comparable.")

        sensitive = set(self._metadata.sensitive_attributes)

        total_weighted_cells = 0.0
        unchanged_weighted_cells = 0.0

        for column in original.columns:
            weight = 2.0 if column in sensitive else 1.0
            original_values = original[column]
            working_values = working[column]

            matches = (original_values == working_values).fillna(False)
            total_weighted_cells += weight * len(matches)
            unchanged_weighted_cells += weight * matches.sum()

        if total_weighted_cells == 0:
            return 100.0, 0.0

        precision_pct = (unchanged_weighted_cells / total_weighted_cells) * 100.0
        privacy_pct = 100.0 - precision_pct
        return precision_pct, privacy_pct

    def get_column_statistics(self, use_working: bool = True) -> pd.DataFrame:
        """Compute main statistics per column for the original or working dataset.

        Stats depend on column type (numeric vs categorical). All columns: count,
        unique_count (number of distinct values), nulls. Numeric: plus mean, min, max,
        std, median. Categorical columns have no extra stats. Columns not in metadata
        are treated as categorical.

        Parameters
        ----------
        use_working:
            If True, use the working dataset; if False, use the original.

        Returns
        -------
        pd.DataFrame:
            One row per column; columns are stat names. Non-applicable stats are NaN.
        """
        df = self.get_working_dataset() if use_working else self.get_original_dataset()
        meta = self._metadata
        rows: List[Dict[str, Any]] = []
        for col in df.columns:
            s = df[col]
            dtype = meta.column_types.get(col, "categorical")
            row: Dict[str, Any] = {
                "column": col,
                "type": dtype,
                "count": int(s.count()),
                "unique_count": int(s.nunique()),
                "nulls": int(s.isna().sum()),
            }
            if dtype == "numeric":
                num = pd.Series(pd.to_numeric(s, errors="coerce"), index=s.index)
                row["mean"] = float(num.mean()) if num.notna().any() else None
                row["min"] = float(num.min()) if num.notna().any() else None
                row["max"] = float(num.max()) if num.notna().any() else None
                row["std"] = float(num.std()) if num.notna().any() else None
                row["median"] = float(num.median()) if num.notna().any() else None
            else:
                row["mean"] = None
                row["min"] = None
                row["max"] = None
                row["std"] = None
                row["median"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    def apply_k_anonymity(
        self,
        k: int,
        bins_start: int = 10,
    ) -> dict:
        """Apply k-anonymity by iteratively generalizing quasi-identifier columns.

        K-anonymity ensures that each combination of quasi-identifier values appears
        in at least *k* rows. Only quasi-identifier columns are modified; the rest of
        the working dataset keeps its current state.

        Quasi-identifiers are restored from the original dataset before applying the
        generalization, so previous generalizations on those columns are overwritten.

        Strategy per column type
        ------------------------
        - **Numeric**: applies ``generalize_numeric_column`` starting with *bins_start*
          bins, halving them each iteration down to a minimum of 2.
        - **Categorical**: truncates one more trailing character per iteration and
          replaces it with ``*`` (e.g. "28001" → "2800*" → "280**").

        Parameters
        ----------
        k:
            Minimum group size. Must be >= 2.
        bins_start:
            Initial number of bins for numeric quasi-identifiers. Must be >= 2.

        Returns
        -------
        dict with keys:
            ``achieved`` (bool): whether k-anonymity was reached.
            ``min_group_size`` (int): smallest group size in the final dataset.
            ``iterations`` (int): number of generalization rounds applied.

        Raises
        ------
            ValueError:
                If k < 2, bins_start < 2, no quasi-identifiers are defined, or only one
                distinct quasi-identifier combination exists.
        """
        if k < 2:
            raise ValueError("k must be >= 2.")
        if bins_start < 2:
            raise ValueError("bins_start must be >= 2.")

        quasi_cols = list(self._metadata.quasi_identifiers)
        if not quasi_cols:
            raise ValueError("No quasi-identifiers are defined in the metadata.")

        # Distinct QI combinations in the current working dataset
        n_distinct = self._working_df[quasi_cols].drop_duplicates().shape[0]
        if n_distinct <= 1:
            raise ValueError(
                "Only one distinct combination of quasi-identifier values exists. "
                "K-anonymity cannot be applied."
            )

        numeric_quasi = [c for c in quasi_cols if self._metadata.column_types.get(c) == "numeric"]
        categorical_quasi = [c for c in quasi_cols if c not in numeric_quasi]

        # Baseline: original values for QI columns (generalisation always starts fresh)
        qi_baseline: Dict[str, Any] = {c: self._original_df[c].copy() for c in quasi_cols}

        # Generalisation state
        current_bins: Dict[str, int] = {c: bins_start for c in numeric_quasi}
        cat_prefix_len: Dict[str, int] = {}
        for c in categorical_quasi:
            max_len = int(qi_baseline[c].dropna().astype(str).str.len().max() or 1)
            cat_prefix_len[c] = max_len  # full length = no truncation yet

        def _apply_current_generalisation() -> None:
            """Restore QI baseline then apply current generalisation settings."""
            for col in quasi_cols:
                self._working_df[col] = qi_baseline[col].copy()

            for col in numeric_quasi:
                self.generalize_numeric_column(col, bins=current_bins[col], include_lowest=True)

            for col in categorical_quasi:
                plen = cat_prefix_len[col]

                def _truncate(val: Any, _plen: int = plen) -> Any:
                    if pd.isna(val):
                        return val
                    s = str(val)
                    return s[:_plen] + "*" * max(0, len(s) - _plen)

                self._working_df[col] = self._working_df[col].apply(_truncate)

        def _min_group_size() -> int:
            counts = self._working_df.groupby(quasi_cols, dropna=False).size()
            return int(counts.min())

        def _can_reduce_more() -> bool:
            return any(current_bins[c] > 2 for c in numeric_quasi) or any(
                cat_prefix_len[c] > 1 for c in categorical_quasi
            )

        iterations = 0

        while True:
            _apply_current_generalisation()
            iterations += 1

            min_g = _min_group_size()
            if min_g >= k:
                return {"achieved": True, "min_group_size": min_g, "iterations": iterations}

            if not _can_reduce_more():
                return {"achieved": False, "min_group_size": min_g, "iterations": iterations}

            # Reduce generalisation settings for next iteration
            for col in numeric_quasi:
                if current_bins[col] > 2:
                    current_bins[col] = max(2, current_bins[col] // 2)
            for col in categorical_quasi:
                if cat_prefix_len[col] > 1:
                    cat_prefix_len[col] -= 1

    def _select_identifiers(self, identifier_column: Optional[str]) -> List[str]:
        """Return the identifier column(s) to operate on by name."""
        identifiers = self._metadata.identifiers
        if not identifiers:
            raise ValueError("No identifiers are configured in metadata.")

        if identifier_column is None:
            return list(identifiers)

        if identifier_column not in identifiers:
            raise ValueError(
                f"Column '{identifier_column}' is not in metadata.identifiers: {identifiers}."
            )
        return [identifier_column]

    def _validate_metadata_consistency(self) -> None:
        """Validate that metadata is consistent with the actual dataset."""
        defined_columns = set(self._original_df.columns)

        for name in self._metadata.column_types:
            if name not in defined_columns:
                raise ValueError(f"Metadata refers to unknown column '{name}'.")

        all_referenced: Iterable[str] = (
            list(self._metadata.identifiers)
            + list(self._metadata.quasi_identifiers)
            + list(self._metadata.sensitive_attributes)
        )
        for column in all_referenced:
            if column not in defined_columns:
                raise ValueError(f"Metadata role refers to unknown column '{column}'.")

    def _ensure_column_exists(self, column: str) -> None:
        """Ensure that the given column exists in the working dataset."""
        if column not in self._working_df.columns:
            raise KeyError(f"Column '{column}' does not exist in the dataset.")

    def _ensure_numeric_column(self, column: str) -> None:
        """Ensure that the given column is numeric according to metadata."""
        declared_type = self._metadata.column_types.get(column)
        if declared_type != "numeric":
            raise TypeError(
                f"Column '{column}' is not declared as numeric in metadata (got '{declared_type}')."
            )

