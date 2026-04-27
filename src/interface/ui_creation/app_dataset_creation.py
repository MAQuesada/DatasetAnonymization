"""Streamlit application for dataset generation"""

import streamlit as st
from dataset_creation import generate_dataset

st.set_page_config(page_title="Dataset Generator", layout="wide")

# ================= SESSION STATE =================
if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "custom_columns" not in st.session_state:
    st.session_state.custom_columns = []

# ================= PAGE =================
st.title("📊 Dataset Generator")

st.subheader("Configuration")

# INPUTS
n_samples = st.number_input("Number of Samples", min_value=1, step=1)

# ---------PRINCIPAL COLUMNS---------------
st.markdown("### Select columns")
col1, col2, col3 = st.columns(3)

with col1:
    name = st.checkbox("Name")
    age = st.checkbox("Age")

with col2:
    surname1 = st.checkbox("First Surname")
    gender = st.checkbox("Gender")

with col3:
    surname2 = st.checkbox("Second Surname")
    municipality = st.checkbox("Municipality of Malaga")

f = st.slider("Random frequency (f)", 0.0, 1.0, 0.1)

columns = {
    "name": name,
    "surname1": surname1,
    "surname2": surname2,
    "age": age,
    "gender": gender,
    "municipality": municipality,
}

# -------------CUSTOM COLUMNS-------------------------
st.markdown("---")
st.subheader("➕ Custom Columns")

col_type = st.selectbox("Column type", ["categorical", "numerical"])

col_name = st.text_input("Column name")

custom_col = None

if col_type == "categorical":
    values = st.text_input("Possible values (comma separated)")
    weights = st.text_input("Weights (optional, comma separated)")

    if values:
        values_list = [v.strip() for v in values.split(",")]

        weights_list = None
        if weights:
            try:
                weights_list = [float(w.strip()) for w in weights.split(",")]
            except:
                st.warning("Weights must be numeric")

        custom_col = {
            "type": "categorical",
            "name": col_name,
            "values": values_list,
            "weights": weights_list,
        }

elif col_type == "numerical":
    low = st.number_input("Min value", value=0.0)
    high = st.number_input("Max value", value=100.0)

    distribution = st.selectbox(
        "Distribution",
        ["normal", "uniform", "binomial", "lognormal"],
    )

    custom_col = {
        "type": "numerical",
        "name": col_name,
        "low": low,
        "high": high,
        "distribution": distribution,
    }

# ADD COLUMN BUTTON
if st.button("Add custom column"):
    if not col_name:
        st.warning("Column name required")
    elif custom_col:
        st.session_state.custom_columns.append(custom_col)
        st.success(f"Added column: {col_name}")

# SHOW CURRENT CUSTOM COLUMNS
if st.session_state.custom_columns:
    st.markdown("### Current custom columns:")

    for i, col in enumerate(st.session_state.custom_columns):
        c1, c2 = st.columns([3, 1])

        with c1:
            st.write(f"**{col['name']}** ({col['type']})")

        with c2:
            if st.button("DELETE", key=f"delete_{i}"):
                st.session_state.custom_columns.pop(i)
                st.rerun()

    if st.button("Clear all custom columns"):
        st.session_state.custom_columns = []
        st.rerun()

st.markdown("---")


# VALIDATION
valid = n_samples > 0 and any(columns.values() or st.session_state.custom_columns)

# BUTTON
if st.button("Generate dataset", disabled=not valid):
    with st.spinner("Generating dataset..."):
        progress_bar = st.progress(0)

        def update_progress(p):
            progress_bar.progress(p)

        df = generate_dataset(
            n_samples,
            columns,
            f,
            custom_columns=st.session_state.custom_columns,
            progress_callback=update_progress,
        )
        st.session_state.dataset = df

    st.switch_page("pages/result.py")
