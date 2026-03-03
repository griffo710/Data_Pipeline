from pipeline.ingest import ingest_data
from pipeline.validate import (
    validate_data,
    validate_missing_values,
    validate_ranges,
    validate_type,
)
from pipeline.cleaning import MissingValueImputer, DuplicateRemover
from pathlib import Path
import yaml
from packages.logging_setup import setup_logging
from packages.save_file import save_clean_data


def main():
    logger, run_id = setup_logging()
    logger.info(f"[{run_id}] Pipeline execution started.")

    try:
        with open("config/schema.yaml", "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"[{run_id}] Configuration loaded succesfully.")

        df = ingest_data(
            r"C:\Users\USER-1\Downloads\Crop_Recommender\Crop_recommendation.csv"
        )
        logger.info(f"[{run_id}] Data ingestion complete. Shape: {df.shape}")

        logger.info(f"[{run_id}] Starting data validation.")

        validate_data(df, config["required_columns"])
        df = validate_type(df, config["column_data_types"])
        validate_missing_values(df, config["missing_threshold"])
        validate_ranges(df, config["ranges"])

        logger.info(f"[{run_id}] Data Validation Complete")

        logger.info(f"[{run_id}] Starting data cleaning.")

        imputer = MissingValueImputer(
            numerical_strategy=config["numerical_imputation"],
            categorical_strategy=config["categorical_imputation"],
            date_strategy=config["date_imputation"],
        )

        deduplicator = DuplicateRemover()

        df = imputer.fit(df).transform(df, logger=logger)
        logger.info(f"[{run_id}] Missing values imputation complete.")

        df = deduplicator.fit_transform(df)
        logger.info(
            f"[{run_id}] Duplicate removal complete. New shape: {df.shape}. "
            f"Rows removed: {deduplicator.removed_count_}"
        )

        output_path = Path("data/processed/clean_data.csv")
        save_clean_data(df, output_path, logger, run_id)

        logger.info(f"[{run_id}] Clean data saved to {output_path.resolve()}")
        logger.info(f"[{run_id}] Pipeline execution completed successfully.")
        logger.info(
            f"[{run_id}] Remaining missing values:\n%s", df.isna().sum().to_string()
        )
        logger.info(f"[{run_id}] Final dataframe schema:\n%s", df.dtypes.to_string())

    except Exception as e:
        logger.error("Pipeline execution failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
