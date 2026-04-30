import streamlit as st

st.set_page_config(page_title="Dataset Result", layout="wide")

# ================== VALIDATION ==================
# If user enters directly without dataset → redirect
if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.switch_page("app_dataset_creation.py")

df = st.session_state.dataset

st.title("📄 Generated Dataset")

# ================== PAGINATION ==================
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

page_size = 10
start = st.session_state.page_idx * page_size
end = start + page_size

st.dataframe(df.iloc[start:end], width="stretch")

# ================== NAVIGATION ==================
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅ Previous") and st.session_state.page_idx > 0:
        st.session_state.page_idx -= 1
        st.rerun()

with col3:
    if st.button("Next ➡") and end < len(df):
        st.session_state.page_idx += 1
        st.rerun()

# ================== INFO ==================
st.markdown(f"""
Showing rows **{start + 1} - {min(end, len(df))}** of **{len(df)}**
""")

st.markdown("---")

# ================== ACTIONS ==================
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Go back to generate again"):
        st.session_state.dataset = None
        st.session_state.page_idx = 0
        st.switch_page("app_dataset_creation.py")

with col2:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="dataset.csv",
        mime="text/csv",
    )