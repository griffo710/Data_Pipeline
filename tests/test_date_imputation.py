import pandas as pd
from pipeline.cleaning import MissingValueImputer


def test_date_imputation_mode():
    # Arrange
    df = pd.DataFrame({"Date": ["01/02/2024", "02/01/2024", "02/01/2024", None, "01/02/2024"]})

    # Convert exactly like validator does
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    imputer = MissingValueImputer(date_strategy="mode")

    # Act
    result = imputer.fit_transform(df)

    # Assert
    assert result["Date"].isna().sum() == 0
    assert result.loc[2, "Date"] == pd.Timestamp("2024-01-02") 
