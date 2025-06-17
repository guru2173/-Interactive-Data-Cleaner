import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import io

def find_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()

def find_duplicates(df: pd.DataFrame) -> pd.Series:
    return df.duplicated(keep=False)

def find_constant_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

def main():
    st.set_page_config(page_title="Interactive Data Cleaner", layout="wide")
    st.title("🧹 Interactive Data Cleaner")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded:
        st.info("Please upload a CSV or Excel file to begin.")
        return

    try:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.subheader("Original Data Preview")
    st.write(df.head())

    missing = find_missing(df)
    dup_mask = find_duplicates(df)
    const_cols = find_constant_cols(df)

    st.sidebar.header("Detected Issues")
    st.sidebar.write(f"Columns with missing values: {len(missing[missing > 0])}")
    st.sidebar.write(f"Duplicate rows: {dup_mask.sum()}")
    st.sidebar.write(f"Constant columns: {len(const_cols)}")

    st.header("🔧 Cleaning Options")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if missing.sum() > 0:
        st.markdown("### Missing Values Handling")
        for col, cnt in missing.items():
            if cnt > 0:
                st.write(f"Column **{col}** has **{cnt}** missing values.")
                action = st.radio(
                    f"What to do with {col}?",
                    ("Drop rows", "Simple Impute", "KNN Impute", "Leave as is"),
                    key=f"missing_{col}"
                )

                if action == "Drop rows":
                    df = df[df[col].notna()]
                elif action == "Simple Impute":
                    strat = "most_frequent" if col not in numeric_cols else st.selectbox(
                        f"Strategy for {col}", ["mean", "median", "most_frequent"], key=f"strat_{col}"
                    )
                    imp = SimpleImputer(strategy=strat)
                    df[[col]] = imp.fit_transform(df[[col]])
                elif action == "KNN Impute":
                    if col not in numeric_cols:
                        st.warning(f"KNN Imputer only works with numeric columns. Skipping '{col}'.")
                        continue
                    k = st.slider(f"K (neighbors) for {col}", 1, 10, 3, key=f"k_{col}")
                    numeric_df = df[numeric_cols]
                    knn_imp = KNNImputer(n_neighbors=k)
                    imputed = knn_imp.fit_transform(numeric_df)
                    df[numeric_cols] = pd.DataFrame(imputed, columns=numeric_cols)
    else:
        st.info("No missing values detected.")

    if dup_mask.sum() > 0:
        st.markdown("### Duplicate Rows Handling")
        if st.checkbox(f"Drop {dup_mask.sum()} duplicate rows?", key="drop_dups"):
            df = df.loc[~dup_mask]
            st.success("Duplicate rows dropped.")
    else:
        st.info("No duplicate rows detected.")

    if const_cols:
        st.markdown("### Constant Columns Handling")
        if st.checkbox(f"Drop constant columns: {const_cols}", key="drop_consts"):
            df = df.drop(columns=const_cols)
            st.success("Constant columns dropped.")
    else:
        st.info("No constant columns detected.")

    st.header("✅ Cleaned Data Preview")
    st.write(df.head())
    st.write(f"Cleaned Data Shape: {df.shape}")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name='cleaned_data.csv',
        mime='text/csv'
    )

    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='CleanedData')
    towrite.seek(0)
    st.download_button(
        label="Download Cleaned Excel",
        data=towrite,
        file_name='cleaned_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

if __name__ == "__main__":
    main()
