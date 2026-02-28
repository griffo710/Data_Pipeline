import pandas as pd


def validate_data(df: pd.DataFrame, required_cols: list):
    missing_columns = set(required_cols) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Some columns are missing:\n {missing_columns}")


def validate_type(df: pd.DataFrame, column_data_types: dict):
    """
    Ensures columns match expected data types.

    For numeric columns:
        - Attempts safe conversion using pandas.to_numeric()
        - Raises error if conversion fails

    For categorical columns:
        - Converts to string type

    Returns
    -------
    DataFrame with corrected types.
    """
    df = df.copy()
    for col, expected_type in column_data_types.items():

        if col not in df.columns:
            raise ValueError(f"{col} not in dataframe.")

        if expected_type == "numeric":
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.isna().sum() > df[col].isna().sum():
                raise ValueError(f"Column '{col}' contains non-numeric values")

            df[col] = converted

        elif expected_type == "categorical":
            # This is to accommodate categorical columns already in binary format.
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype("Int64").astype("string")
            else:
                df[col] = df[col].astype("string")

        elif expected_type == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

        else:
            raise ValueError(f"Unknown expected type for {col}")

    return df


def validate_missing_values(df: pd.DataFrame, threshold: float):
    missing_ratio = df.isna().mean()

    bad_cols = missing_ratio[missing_ratio > threshold]

    if not bad_cols.empty:
        raise ValueError(f"Columns exceed missing threshold: {bad_cols.to_dict()}")


def validate_ranges(df, ranges):
    for col, (min_val, max_val) in ranges.items():
        series = df[col].dropna()
        if not series.between(min_val, max_val).all():
            raise ValueError(f"{col} has values outside the range")
