import pandas as pd
from pathlib import Path


def ingest_data(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    if path.suffix == ".csv":
        return pd.read_csv(path)

    elif path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    else:
        raise ValueError(
            f"Only excel and csv files are allowed in this pipeline: Yours is a {path.suffix}"
        )
