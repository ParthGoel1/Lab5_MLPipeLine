# Lab 5 - ML Pipeline

This lab builds a complete machine learning workflow using the Titanic dataset.

The main focus was not just training a model, but making sure the data was handled properly before modeling. I first explored the dataset, checked missing values and leakage, documented cleaning decisions, created a fixed train/test split, and then used pipelines for preprocessing and model comparison.

## Files

- `data.py` - provided dataset loader. It tries to load the Titanic dataset from OpenML and falls back to a synthetic version if needed.
- `pipeline.py` - contains the feature engineering and sklearn pipeline helper functions.
- `day1_exploration.ipynb` - data exploration, missingness checks, cleaning decisions, leakage checks, and train/test split.
- `day2_pipeline_modeling.ipynb` - preprocessing, feature engineering, cross-validation, model comparison, tuning, and final test evaluation.
- `split.joblib` - saved train/test split from Day 1 so Day 2 uses the exact same split.
- `requirements.txt` - Python packages needed for the lab.

## Main Workflow

### Day 1

Day 1 is mainly about understanding the dataset before fitting anything.

The notebook includes:

- loading the Titanic data
- checking column types and missing values
- checking whether missingness itself is related to survival
- looking for unusual or invalid values
- identifying leakage columns such as `boat` and `body`
- documenting cleaning decisions
- creating `X` and `y`
- splitting the data using `stratify=y`
- saving the split to `split.joblib`

No scaler, imputer, encoder, or model should be fitted before the train/test split.

### Day 2

Day 2 continues from the saved split.

The notebook includes:

- loading `split.joblib`
- feature engineering
- separating numeric and categorical columns
- building a `ColumnTransformer`
- building a full sklearn `Pipeline`
- cross-validating a Logistic Regression baseline
- comparing multiple model types
- tuning a model using `GridSearchCV`
- evaluating the test set only at the end

## Feature Engineering

The current `engineer()` function creates:

- `has_cabin`
- `age_missing`
- `title`
- `family_size`

The raw `Name` and `Cabin` columns are removed after the useful information is extracted.

## Preprocessing

Numeric columns use:

- median imputation
- `StandardScaler`

Categorical columns use:

- most-frequent imputation
- `OneHotEncoder(handle_unknown="ignore")`

`Pclass` is manually treated as a categorical feature even though it is stored as a number.

## How to Run

Create or activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Make sure all the lab files are in the same folder.

Example folder structure:

```text
lab5_ml_pipeline/
├── data.py
├── pipeline.py
├── day1_exploration.ipynb
├── day2_pipeline_modeling.ipynb
├── requirements.txt
└── README.md
```

Then run the notebooks in this order.

### 1. Run Day 1

Open:

```text
day1_exploration.ipynb
```

Restart the kernel and run all cells from top to bottom.

At the end, the notebook should create:

```text
split.joblib
```

### 2. Run Day 2

Open:

```text
day2_pipeline_modeling.ipynb
```

Restart the kernel and run all cells from top to bottom.

Day 2 should load the saved split instead of creating a new one.

## Important Notes

The test set should stay untouched until the final evaluation.

Cross-validation and `GridSearchCV` should only use the training data.

If OpenML is unavailable, `data.py` automatically uses the provided synthetic Titanic-style dataset, so the notebooks should still run.

The exact model scores may be slightly different depending on whether the real or fallback dataset is used.

## Final Result

On my original run using the real Titanic data:

- baseline Logistic Regression test F1: `0.790`
- tuned Logistic Regression test F1: `0.786`

The tuned model did not improve the final test score. The difference was very small, so I reported it as-is instead of trying to tune again using the test result.

## Main Takeaway

The biggest takeaway from this lab was that getting the pipeline and evaluation process right matters as much as the model itself.

A slightly higher cross-validation score does not always mean a model is clearly better, especially when the difference is smaller than the variation between folds. Tuning also does not guarantee better performance on unseen data.
