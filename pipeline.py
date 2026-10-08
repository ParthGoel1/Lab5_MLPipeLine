import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def engineer(df):
    df = df.copy()

    # cabin info might matter even if the actual cabin number is too messy
    df["has_cabin"] = df["Cabin"].notna().astype(int)

    # missing age itself seemed related to survival
    df["age_missing"] = df["Age"].isna().astype(int)

    # title may be more useful than the full name
    df["title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

    # combine family-related columns
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    df = df.drop(columns=["Name", "Cabin"])

    return df


def get_column_groups(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Pclass is categorical even though it is stored as a number
    if "Pclass" in num_cols:
        num_cols.remove("Pclass")
        cat_cols.append("Pclass")

    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return pre


def build_pipeline(num_cols, cat_cols, model):

    pre = build_preprocessor(num_cols, cat_cols)

    return Pipeline([
        ("preprocessor", pre),
        ("model", model)
    ])