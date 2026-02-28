import pandas as pd
from pipeline.cleaning import DuplicateRemover


def test_remove_duplicates():
    data = pd.DataFrame(
        {
            "ID": [1, 2, 3, 1],
            "Name": ["Mark", "Mary", "Ashley", "Mark"],
            "Gender": ["Male", "Male", "Female", "Male"],
            "Age": [23, 67, 55, 23],
        }
    )

    deduplicator = DuplicateRemover()

    result = deduplicator.fit_transform(data)
    assert data.shape != result.shape
    assert result.shape == (3, 4)
