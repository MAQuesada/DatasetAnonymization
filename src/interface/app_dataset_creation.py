"""Streamlit application for dataset generation"""

import streamlit as st
from dataset_creation import generate_dataset

st.set_page_config(page_title="Dataset Generator", layout="wide")

# ================= SESSION STATE =================
if "dataset" not in st.session_state:
    st.session_state.dataset = None


# ================= PAGE =================
st.title("📊 Dataset Generator")

st.subheader("Configuration")

# INPUTS
n_samples = st.number_input("Number of Samples", min_value=1, step=1)

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

# VALIDATION
valid = n_samples > 0 and any(columns.values())

# BUTTON
if st.button("Generate dataset", disabled=not valid):
    with st.spinner("Generating dataset..."):
        df = generate_dataset(n_samples, columns, f)
        st.session_state.dataset = df

    st.switch_page("pages/result.py")