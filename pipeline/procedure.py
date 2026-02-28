from sklearn.pipeline import Pipeline
from cleaning import MissingValueImputer, DuplicateRemover

model_pipeline = Pipeline(
    steps=[
        ("deduplicate", DuplicateRemover()),
        (
            "impute_missing",
            MissingValueImputer(
                numerical_strategy="mean",
                categorical_strategy="mode",
                date_strategy="mode",
            ),
        ),
    ]
)
