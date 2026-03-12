def detect_data_issues(df):

    issues = []

    for col in df.columns:

        missing_ratio = df[col].isnull().mean()

        if missing_ratio > 0.3:
            issues.append(f"{col} has more than 30% missing values")

        if df[col].nunique() == 1:
            issues.append(f"{col} contains only one unique value")

    return issues