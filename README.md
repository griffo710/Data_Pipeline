# Data Processing Pipeline 

A modular, testable, and reproducible data processing pipeline built in
Python using a scikit-learn compatible transformer architecture.


------------------------------------------------------------------------

## 🚀 Key Features

### 1️⃣ Structured Logging System

-   Logs to console + timestamped file
-   Unique pipeline run ID for traceability
-   Tracks:
    -   Data ingestion results
    -   Validation outcomes
    -   Imputation statistics
    -   Duplicate removal summary
    -   Final dataset schema

### 2️⃣ Scikit-Learn Compatible Transformers

Custom transformers follow the sklearn contract: - `fit()` learns
statistics - `transform()` applies them - Safe for pipelines and
production use

Implemented transformers: - MissingValueImputer - DuplicateRemover

### 3️⃣ Data Validation Layer

Validates: - Required columns - Data types - Value ranges - Missing
value thresholds

### 4️⃣ Reproducible Pipeline Execution

-   Configuration-driven behavior
-   Deterministic transformations
-   No data leakage

### 5️⃣ Automated Testing

-   Unit tests using pytest
-   CI-ready structure
-   Tests for:
    -   Numerical imputation
    -   Categorical imputation
    -   Date imputation
    -   Error handling

------------------------------------------------------------------------

## 🧱 Project Structure

    Data_Pipeline/
    │
    ├── pipeline/
    │   ├── ingest.py
    │   ├── validate.py
    │   └── cleaning.py
    │
    ├── packages/
    │   ├── logging_setup.py
    │   └── save_file.py
    │
    ├── config/
    │   └── schema.yaml
    │
    ├── tests/
    |   ├── conftest.py
    │   ├── test_imputer.py
    │   ├── test_categorical_imputation.py
    |   ├── test_removing_duplicates.py
    |   ├── test_undefined_measure.py
    |   ├── test_empty_dataset.py
    │   └── test_date_imputation.py
    │
    ├── data/processed
    │   └── processed/
    │
    ├── logs/
    ├── main.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Clone repository

    git clone <your-repo-url>
    cd Data_Pipeline

### 2️⃣ Create virtual environment

    python -m venv venv

Activate:

Windows:

    venv\Scripts\activate

Linux / Mac:

    source venv/bin/activate

### 3️⃣ Install dependencies

    python -m pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Running the Pipeline

    python main.py

Pipeline will:

1.  Load configuration
2.  Ingest dataset
3.  Validate schema and values
4.  Impute missing values
5.  Remove duplicates
6.  Save clean dataset
7.  Generate structured logs

Output:

    data/processed/clean_data.csv
    logs/<timestamp>.log

------------------------------------------------------------------------

## 🧪 Running Tests

    pytest tests

Tests validate correctness of transformations and error handling.

------------------------------------------------------------------------

## 📌 Design Principles

### ✔ Reproducibility

All transformations are deterministic and configuration-driven.

### ✔ No Data Leakage

Statistics are learned only during `fit()`.

### ✔ Observability

Pipeline behavior is fully traceable through structured logging.

### ✔ Production-Oriented Architecture

Modular components mirror real ML data pipelines.

------------------------------------------------------------------------

## 📈 Future Improvements

-   Pipeline orchestration
-   Feature engineering module
-   Model training stage
-   Docker containerization
-   Cloud deployment
-   Data versioning

------------------------------------------------------------------------
