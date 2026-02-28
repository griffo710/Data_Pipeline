import pandas as pd
import numpy as np
from pipeline.cleaning import MissingValueImputer


def test_numeric_imputation_mean():
    # Arrange (create controlled data)
    df = pd.DataFrame({"A": [1, 2, None, 4]})

    imputer = MissingValueImputer(numerical_strategy="mean")

    # Act
    result = imputer.fit_transform(df)

    # Assert
    expected_mean = (1 + 2 + 4) / 3
    assert result["A"].isna().sum() == 0
    assert np.isclose(result.loc[2, "A"], expected_mean, atol=1e-2)
