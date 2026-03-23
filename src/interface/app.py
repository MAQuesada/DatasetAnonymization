"""Streamlit application for dataset anonymization."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure dataset_anonymization is importable (e.g. when running: streamlit run src/interface/app.py)
_app_file = Path(__file__).resolve()
_src_dir = _app_file.parent.parent  # src/interface -> src
_project_root = _src_dir.parent     # src -> project root
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
    """Round numeric stats and use '-' for missing/non-applicable values."""
    out = stats_df.copy()
    for col in ["mean", "min", "max", "std", "median"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: "-" if pd.isna(x) else round(float(x), 4)
            )
    return out


def _style_data_by_role(df: pd.DataFrame, metadata: DatasetMetadata) -> pd.io.formats.style.Styler:
    """Color columns based on metadata role."""
    identifiers = set(metadata.identifiers)
    quasi_identifiers = set(metadata.quasi_identifiers)
    sensitive = set(metadata.sensitive_attributes)

    def _column_style(col_name: str) -> str:
        # Priority if a column appears in multiple lists: sensitive > identifier > quasi.
        if col_name in sensitive:
            return "background-color: #fff5f5; color: #1f2937;"  # very soft red + dark text
        if col_name in identifiers:
            return "background-color: #f3fbf5; color: #1f2937;"  # very soft green + dark text
        if col_name in quasi_identifiers:
            return "background-color: #fffdf2; color: #1f2937;"  # very soft yellow + dark text
        return ""

    style_map = {col: _column_style(col) for col in df.columns}
    return df.style.apply(lambda _: [style_map[c] for c in df.columns], axis=1)


def _render_data_view(manager: DatasetManager, n_rows: int, dataset_key: str) -> None:
    view = st.radio(
        "View", ["Original", "Modified"], horizontal=True, key=f"view_toggle_{dataset_key}"
    )
    df = manager.get_original_dataset() if view == "Original" else manager.get_working_dataset()
    display = df.head(n_rows)
    styled_display = _style_data_by_role(display, manager.metadata)
    st.dataframe(styled_display, use_container_width=True, key=f"dataframe_{dataset_key}")
    st.caption(f"Showing first {n_rows} of {len(df)} rows. Columns: {list(df.columns)}")
    st.caption("Legend: identifiers = green, quasi-identifiers = yellow, sensitive attributes = red.")

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
    all_cols = list(manager.get_working_dataset().columns)

    st.subheader("De-identification")
    id_col_opt = [None] + identifiers if identifiers else []
    id_choice = st.selectbox(
        "Identifier column",
        options=id_col_opt,
        format_func=lambda x: "All identifiers" if x is None else x,
        key="deid_col",
    )
    col1, col2 = st.columns(2)
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
    if manager.ope_mapping_columns:
        if st.button("Reverse OPE pseudonyms"):
            try:
                manager.reverse_order_preserving_pseudonyms(identifier_column=id_choice)
                st.success("Reversed.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.subheader("Numeric anonymization")
    if not numeric_cols:
        st.caption("No numeric columns in metadata.")
    else:
        num_col = st.selectbox("Numeric column", numeric_cols, key="num_col")
        tab_gen, tab_pert = st.tabs(["Generalize", "Perturb"])
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
            if st.button("Apply perturbation"):
                try:
                    manager.perturb_numeric_column(num_col, noise_std=noise_std, random_state=random_state)
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if st.button("Reset working dataset to original"):
        manager.reset_working_dataset()
        st.success("Reset.")
        st.rerun()

    st.subheader("Masking")
    if not all_cols:
        st.caption("No columns available for masking.")
    else:
        mask_col = st.selectbox("Column to mask", all_cols, key="mask_col")
        mask_strategy = st.radio(
            "Masking strategy",
            options=[
                "mask_first_k",
                "mask_last_k",
                "mask_before_first_space",
                "mask_after_last_space",
                "mask_regex",
            ],
            format_func=lambda s: {
                "mask_first_k": "Mask first K characters",
                "mask_last_k": "Mask last K characters",
                "mask_before_first_space": "Mask before first space",
                "mask_after_last_space": "Mask after last space",
                "mask_regex": "Advanced: mask by regular expression",
            }[s],
            key="mask_strategy",
        )

        k_value: int | None = None
        regex_value: str | None = None

        if mask_strategy in {"mask_first_k", "mask_last_k"}:
            k_value = int(
                st.number_input(
                    "K",
                    min_value=0,
                    value=1,
                    step=1,
                    key="mask_k",
                )
            )

        if mask_strategy == "mask_regex":
            regex_value = st.text_input(
                "Regex pattern",
                value=r"\d",
                key="mask_regex_pattern",
                help="Example: \\d masks every digit. Example: [AEIOUaeiou] masks vowels.",
            )
            st.caption(
                "Regex example: `\\d` -> replaces all digits with `*`."
            )

        if st.button("Apply masking"):
            try:
                if mask_col is None:
                    raise ValueError("Select a column to mask.")
                manager.mask_column(
                    column=mask_col,
                    strategy=mask_strategy,
                    k=k_value,
                    regex_pattern=regex_value,
                )
                st.success("Masking applied.")
                st.rerun()
            except Exception as e:
                st.error(str(e))


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
