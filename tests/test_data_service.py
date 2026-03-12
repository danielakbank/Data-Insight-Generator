import pandas as pd
from app.services.data_service import summarize_dataset

def test_summary():

    df = pd.DataFrame({
        "sales":[100,200,300]
    })

    summary = summarize_dataset(df)

    assert summary["metadata"]["rows"] == 3