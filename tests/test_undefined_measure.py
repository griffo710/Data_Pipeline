import pandas as pd
import pytest
from pipeline.cleaning import MissingValueImputer


def test_unknown_central_tendency_measure():
    df = pd.DataFrame({"Gender": ["Male", "Female", "Male"]})

    imputer = MissingValueImputer(categorical_strategy="None")

    with pytest.raises(ValueError):
        imputer.fit_transform(df)
