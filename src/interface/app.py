"""Streamlit application for dataset anonymization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_app_file = Path(__file__).resolve()
_src_dir = _app_file.parent.parent
_project_root = _src_dir.parent
for _dir in (_src_dir, _project_root):
    _dir_str = str(_dir)
    if _dir_str not in sys.path and (_dir / "dataset_anonymization").exists():
        sys.path.insert(0, _dir_str)
        break
else:
    sys.path.insert(0, str(_src_dir))

import pandas as pd
import streamlit as st

from dataset_anonymization import (
    DatasetManager,
    DatasetManagerLoadError,
    DatasetMetadata,
)

DEFAULT_DATASETS_PATH = "dataset_data"


def _ensure_session_state() -> None:
    if "manager" not in st.session_state:
        st.session_state.manager = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None


def _storage_dir() -> Path:
    return Path(DEFAULT_DATASETS_PATH)


def _list_saved_names() -> list[str]:
    dir_path = _storage_dir()
    if not dir_path.is_dir():
        return []
    return sorted(p.stem for p in dir_path.glob("*.pkl"))


def _render_load_from_list() -> None:
    names = _list_saved_names()
    if not names:
        st.warning("No saved datasets yet. Create a new one first.")
        return
    selected = st.selectbox("Saved datasets", options=names, key="load_select")
    if st.button("Load", key="load_btn"):
        try:
            path = _storage_dir() / f"{selected}.pkl"
            manager = DatasetManager.load(path)
            st.session_state.manager = manager
            st.session_state.dataset_name = selected
            st.success(f"Loaded **{selected}**.")
            st.rerun()
        except DatasetManagerLoadError as e:
            st.error(str(e))


def _render_create_new() -> None:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        return
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return
    columns = list(df.columns)
    if not columns:
        st.error("The CSV has no columns.")
        return

    st.subheader("Metadata")
    st.caption("Define column types and roles. All referenced columns must exist in the CSV.")

    col_types: dict[str, str] = {}
    for col in columns:
        col_types[col] = st.selectbox(
            f"Type for **{col}**",
            options=["numeric", "categorical"],
            key=f"type_{col}",
        )

    identifiers = st.multiselect("Identifiers (uniquely identify a person)", options=columns, key="identifiers")
    quasi = st.multiselect("Quasi-identifiers", options=columns, key="quasi")
    sensitive = st.multiselect("Sensitive attributes", options=columns, key="sensitive")

    if st.button("Create dataset"):
        metadata = DatasetMetadata(
            column_types=col_types,
            identifiers=identifiers,
            quasi_identifiers=quasi,
            sensitive_attributes=sensitive,
        )
        try:
            manager = DatasetManager(df, metadata)
            st.session_state.manager = manager
            st.success("Dataset created. You can now view and apply transformations.")
            st.rerun()
        except ValueError as e:
            st.error(f"Validation failed: {e}")
        except Exception as e:
            st.error(str(e))


def _format_stats_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()
    for col in ["mean", "min", "max", "std", "median"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: "-" if pd.isna(x) else round(float(x), 4))
    for col in ["mode", "mode_count"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: "-" if pd.isna(x) else x)
    return out


def _render_data_view(manager: DatasetManager, n_rows: int, dataset_key: str) -> None:
    view = st.radio(
        "View", ["Original", "Modified"], horizontal=True, key=f"view_toggle_{dataset_key}"
    )
    df = manager.get_original_dataset() if view == "Original" else manager.get_working_dataset()
    display = df.head(n_rows)
    st.dataframe(display, use_container_width=True, key=f"dataframe_{dataset_key}")
    st.caption(f"Showing first {n_rows} of {len(df)} rows. Columns: {list(df.columns)}")

    use_working = view == "Modified"
    stats_df = manager.get_column_statistics(use_working=use_working)
    stats_display = _format_stats_table(stats_df)
    st.subheader("Column statistics")
    st.caption(f"Statistics for the **{view.lower()}** dataset.")
    st.dataframe(stats_display, use_container_width=True, key=f"stats_{dataset_key}_{view}")


def _render_actions(manager: DatasetManager) -> None:
    meta = manager.metadata
    identifiers = meta.identifiers
    numeric_cols = [c for c, t in meta.column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in meta.column_types.items() if t == "categorical"]

    # ------------------------------------------------------------------ #
    # De-identification                                                    #
    # ------------------------------------------------------------------ #
    st.subheader("De-identification")
    id_col_opt = [None] + identifiers if identifiers else []
    id_choice = st.selectbox(
        "Identifier column (None = all)",
        options=id_col_opt,
        format_func=lambda x: "All identifiers" if x is None else x,
        key="deid_col",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Random pseudonyms"):
            try:
                manager.deidentify_with_random_pseudonyms(identifier_column=id_choice)
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("Order-preserving pseudonyms (OPE)"):
            try:
                manager.deidentify_with_order_preserving_pseudonyms(identifier_column=id_choice)
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with col3:
        if st.button("HMAC pseudonyms"):
            try:
                manager.deidentify_with_hmac_pseudonyms(identifier_column=id_choice)
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    col_rev1, col_rev2 = st.columns(2)
    with col_rev1:
        if manager.ope_mapping_columns:
            if st.button("Reverse OPE pseudonyms"):
                try:
                    manager.reverse_order_preserving_pseudonyms(identifier_column=id_choice)
                    st.success("Reversed.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with col_rev2:
        if manager.hmac_mapping_columns:
            if st.button("Reverse HMAC pseudonyms"):
                try:
                    manager.reverse_hmac_pseudonyms(identifier_column=id_choice)
                    st.success("Reversed.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # ------------------------------------------------------------------ #
    # Numeric anonymization                                                #
    # ------------------------------------------------------------------ #
    st.subheader("Numeric anonymization")
    if not numeric_cols:
        st.caption("No numeric columns in metadata.")
    else:
        num_col = st.selectbox("Numeric column", numeric_cols, key="num_col")
        tab_gen, tab_pert, tab_laplace = st.tabs(["Generalize", "Perturb (Gaussian)", "Perturb (Laplace)"])
        with tab_gen:
            bins = st.number_input("Bins", min_value=2, value=10, key="bins")
            include_lowest = st.checkbox("Include lowest", value=True, key="include_lowest")
            if st.button("Apply generalization"):
                try:
                    manager.generalize_numeric_column(num_col, bins=bins, include_lowest=include_lowest)
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with tab_pert:
            noise_std = st.number_input("Noise (fraction of range)", min_value=0.01, value=0.05, step=0.01, key="noise_std")
            rs = st.number_input("Random state (optional)", value=-1, min_value=-1, key="rs")
            random_state: int | None = None if rs < 0 else int(rs)
            if st.button("Apply perturbation (Gaussian)"):
                try:
                    manager.perturb_numeric_column(num_col, noise_std=noise_std, random_state=random_state)
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with tab_laplace:
            epsilon = st.number_input("Epsilon (privacy budget)", min_value=0.01, value=1.0, step=0.1, key="epsilon")
            sensitivity = st.number_input("Sensitivity", min_value=0.01, value=1.0, step=0.1, key="sensitivity")
            rs_lap = st.number_input("Random state (optional)", value=-1, min_value=-1, key="rs_lap")
            random_state_lap: int | None = None if rs_lap < 0 else int(rs_lap)
            st.caption("Smaller epsilon = more noise = stronger privacy guarantee.")
            if st.button("Apply perturbation (Laplace)"):
                try:
                    manager.perturb_numeric_column_laplace(
                        num_col, sensitivity=sensitivity, epsilon=epsilon, random_state=random_state_lap
                    )
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # ------------------------------------------------------------------ #
    # Categorical generalization                                           #
    # ------------------------------------------------------------------ #
    st.subheader("Categorical generalization")
    if not categorical_cols:
        st.caption("No categorical columns in metadata.")
    else:
        cat_col = st.selectbox("Categorical column", categorical_cols, key="cat_col")
        st.caption(
            'Enter a mapping as JSON. Example: {"Flu": "Respiratory", "Cold": "Respiratory", "Diabetes": "Metabolic"}'
        )
        mapping_input = st.text_area("Mapping (JSON)", value="{}", key="cat_mapping")
        default_val = st.text_input(
            "Default for unmapped values (leave empty to keep original)", value="", key="cat_default"
        )
        if st.button("Apply categorical generalization"):
            try:
                mapping = json.loads(mapping_input)
                if not isinstance(mapping, dict):
                    st.error("Mapping must be a JSON object.")
                else:
                    default = default_val.strip() if default_val.strip() else None
                    manager.generalize_categorical_column(cat_col, mapping=mapping, default=default)
                    st.success("Done.")
                    st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            except Exception as e:
                st.error(str(e))

    if st.button("Reset working dataset to original"):
        manager.reset_working_dataset()
        st.success("Reset.")
        st.rerun()


def _render_export_and_save(manager: DatasetManager) -> None:
    st.subheader("Save & export")
    name = st.session_state.dataset_name or ""
    save_name = st.text_input("Dataset name (used when saving and exporting)", value=name, key="save_name", placeholder="e.g. Diabetes")
    if st.button("Save"):
        if not save_name.strip():
            st.error("Enter a dataset name.")
        else:
            try:
                storage = _storage_dir()
                storage.mkdir(parents=True, exist_ok=True)
                path = storage / f"{save_name.strip()}.pkl"
                manager.save(path)
                st.session_state.dataset_name = save_name.strip()
                st.success("Dataset saved successfully.")
                st.info(f"**{save_name.strip()}** has been saved. You can load it later from the list.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    export_filename = f"{st.session_state.dataset_name}.csv" if st.session_state.dataset_name else "export.csv"
    st.caption(f"Export modified dataset to CSV as **{export_filename}** (save the dataset first to use its name).")
    export_folder = st.text_input("Export folder", value=DEFAULT_DATASETS_PATH, key="export_folder")
    if st.button("Export to CSV"):
        path = Path(export_folder) / export_filename
        try:
            manager.export_working_to_csv(path, index=False)
            st.success(f"Exported as **{export_filename}**.")
        except Exception as e:
            st.error(str(e))


def main() -> None:
    st.set_page_config(page_title="Dataset Anonymization", layout="wide")
    _ensure_session_state()

    st.title("Dataset Anonymization")
    mode = st.sidebar.radio("Mode", ["Load", "Create new"], key="mode")
    if mode == "Load":
        _render_load_from_list()
    else:
        _render_create_new()

    manager: DatasetManager | None = st.session_state.manager
    if manager is None:
        st.info("Load a dataset from the list or create a new one to continue.")
        return

    dataset_key = st.session_state.dataset_name or f"new_{id(manager)}"
    if st.session_state.dataset_name:
        st.sidebar.caption(f"Current: **{st.session_state.dataset_name}**")
    n_rows = st.sidebar.number_input("Rows to show", min_value=1, value=20, key="n_rows")
    st.sidebar.divider()
    precision, privacy = manager.compute_precision_privacy_tradeoff()
    st.sidebar.metric("Precision %", round(precision, 1))
    st.sidebar.metric("Privacy %", round(privacy, 1))

    _render_data_view(manager, int(n_rows), dataset_key)
    st.divider()
    _render_actions(manager)
    st.divider()
    _render_export_and_save(manager)


if __name__ == "__main__":
    main()