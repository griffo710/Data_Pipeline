import os


def save_clean_data(df, output_path, logger, run_id):
    """
    Save cleaned dataset.

    Ensures directory exists and overwrites existing file.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
