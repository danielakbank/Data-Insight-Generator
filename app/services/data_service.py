import pandas as pd

# ----------------------------
# Dataset Loading & Inspection
# ----------------------------
def load_dataset(file):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file)

def inspect_dataset(df):
    """Return basic metadata about the dataset."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

# ----------------------------
# Numeric Column Detection & Cleaning
# ----------------------------
def detect_numeric_columns_safe(df, threshold=0.9, exclude_id_cols=True):
    """
    Detect numeric-like columns safely, optionally ignoring ID columns.
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
                if "id" in col.lower():
                    continue
                if df[col].nunique() / len(df) > 0.9:
                    continue
            numeric_cols.append(col)

    return numeric_cols

def clean_numeric_columns_safe(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert numeric-like columns safely to numeric, coercing errors to NaN."""
    for col in columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('[^\d\.\-]', '', regex=True),
            errors='coerce'
        )
    return df

# ----------------------------
# Numeric Summary
# ----------------------------
def numeric_summary(df: pd.DataFrame) -> dict:
    """Return standard descriptive stats for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}
    return numeric_df.describe().to_dict()

# ----------------------------
# Categorical Summary
# ----------------------------
def categorical_summary(df: pd.DataFrame, top_n=10) -> dict:
    """Return summary for categorical columns with top N values."""
    cat_cols = df.select_dtypes(exclude=["number"])
    summary = {}

    for col in cat_cols:
        top_values = df[col].value_counts().head(top_n).to_dict()
        summary[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": top_values
        }

    return summary

# ----------------------------
# Correlation Analysis
# ----------------------------
def correlation_analysis(df: pd.DataFrame, threshold=0.7, top_n=10) -> dict:
    """Return strong correlations (|r| >= threshold), limited to top N per column."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return {}

    corr = numeric_df.corr()
    strong = {}

    for col in corr.columns:
        # get top N correlations by absolute value, excluding self
        abs_corr = corr[col].drop(col).abs().sort_values(ascending=False).head(top_n)
        strong[col] = {other: round(float(corr.loc[col, other]), 2) 
                       for other in abs_corr.index if abs_corr[other] >= threshold}

    return strong

# ----------------------------
# Full Dataset Summary
# ----------------------------
def generate_dataset_summary_full(df: pd.DataFrame, exclude_cols=None, top_n=10) -> dict:
    """
    Generate a full summary including:
    - Metadata
    - Numeric summary
    - Categorical summary (top N values)
    - Strong correlations (top N)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Detect numeric columns excluding IDs
    numeric_cols = detect_numeric_columns_safe(df)
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    df = clean_numeric_columns_safe(df, numeric_cols)

    return {
        "metadata": inspect_dataset(df),
        "numeric_summary": numeric_summary(df[numeric_cols]) if numeric_cols else {},
        "categorical_summary": categorical_summary(df, top_n=top_n),
        "correlations": correlation_analysis(df[numeric_cols], top_n=top_n) if numeric_cols else {}
    }