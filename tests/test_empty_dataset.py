import pandas as pd
from pipeline.cleaning import MissingValueImputer
import pytest


def test_error_in_empty_dataset():
    df = pd.DataFrame({"Hours_of_study": [None, None, None, None]})

    imputer = MissingValueImputer(numerical_strategy="mean")

    with pytest.raises(ValueError):
        imputer.fit_transform(df)
