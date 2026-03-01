import pandas as pd
from pipeline.cleaning import MissingValueImputer


def test_impute_categorical_mode():
    data = pd.DataFrame(
        {"Gender": ["Male", "Female", None, "Other", None, "Other", "Male", "Male"]}
    )

    imputer = MissingValueImputer(categorical_strategy="mode")

    result = imputer.fit_transform(data)

    print(result)
    assert result["Gender"].isna().sum() == 0
    assert result.loc[2, "Gender"] == "Male"
