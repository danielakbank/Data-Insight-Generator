# app/services/data_service.py

import pandas as pd


def load_dataset(file):
    return pd.read_csv(file)


def inspect_dataset(df):

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

def detect_numeric_columns_safe(df, threshold=0.9, exclude_id_cols=True):
    """
    Detect numeric-like columns safely, optionally ignoring ID columns.
    
    Parameters:
    - df: pd.DataFrame
    - threshold: fraction of numeric values required
    - exclude_id_cols: whether to skip columns containing 'id' or very high cardinality
    """
    numeric_cols = []

    for col in df.columns:
        col_series = df[col].astype(str)
        cleaned = col_series.str.replace(r"[^\d\.\-]", "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce").notnull()
        ratio = numeric.sum() / len(col_series)

        if ratio >= threshold:
            # Exclude ID-like columns
            if exclude_id_cols:
                # Check column name for 'id'
                if "id" in col.lower():
                    continue
                # Skip high-cardinality numeric columns (likely IDs)
                if df[col].nunique() / len(df) > 0.9:
                    continue
            numeric_cols.append(col)

    return numeric_cols


def clean_numeric_columns_safe(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    for col in columns:

        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('[^\d\.\-]', '', regex=True),
            errors='coerce'
        )

    return df


def numeric_summary(df):

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return {}

    return numeric_df.describe().to_dict()


def categorical_summary(df):

    cat_cols = df.select_dtypes(exclude=["number"])

    summary = {}

    for col in cat_cols:

        top_values = df[col].value_counts().head(5).to_dict()

        summary[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": top_values
        }

    return summary


def correlation_analysis(df, threshold=0.7):

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return {}

    corr = numeric_df.corr()

    strong = {}

    for col in corr.columns:

        for other in corr.columns:

            val = corr.loc[col, other]

            if col != other and abs(val) >= threshold:

                strong.setdefault(col, {})[other] = round(float(val), 2)

    return strong


def generate_dataset_summary_full(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    # detect numeric columns excluding IDs
    numeric_cols = detect_numeric_columns_safe(df)
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    df = clean_numeric_columns_safe(df, numeric_cols)

    return {
        "metadata": inspect_dataset(df),
        "numeric_summary": numeric_summary(df[numeric_cols]) if numeric_cols else {},
        "categorical_summary": categorical_summary(df),
        "correlations": correlation_analysis(df[numeric_cols]) if numeric_cols else {}
    }