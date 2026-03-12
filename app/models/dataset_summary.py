from dataclasses import dataclass

@dataclass
class DatasetSummary:

    metadata: dict
    column_profile: dict
    numeric_summary: dict
    categorical_summary: dict
    correlations: dict
    data_quality_issues: list