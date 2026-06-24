"""Lab 3: Student survey simple linear regression.

This script mirrors the notebook workflow:
- export the Google Forms Excel response file to CSV
- clean CIA, GPA, and attendance values
- run Simple Linear Regression with scikit-learn
- manually compute Ordinary Least Squares parameters
- compare predictions
- save learned parameters with pickle
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Department Awareness Survey (Responses).xlsx"
CSV_FILE = BASE_DIR / "student_survey.csv"
PICKLE_FILE = BASE_DIR / "linear_regression_weights.pkl"

COLUMN_MAP = {
    "your CIA % of last semester ": "CIA_Percentage",
    "your GPA of last semester": "GPA",
    "Your maximum attendance % till last semester": "Attendance_Percentage",
}


def load_raw_survey() -> pd.DataFrame:
    """Load the survey export, preferring CSV once it has been created."""
    if CSV_FILE.exists():
        return pd.read_csv(CSV_FILE)
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            "Expected student_survey.csv or Department Awareness Survey (Responses).xlsx"
        )
    raw_df = pd.read_excel(EXCEL_FILE)
    raw_df.to_csv(CSV_FILE, index=False)
    return raw_df


def export_csv_from_excel() -> pd.DataFrame:
    """Create the required student_survey.csv from the Excel response export."""
    if EXCEL_FILE.exists():
        raw_df = pd.read_excel(EXCEL_FILE)
        raw_df.to_csv(CSV_FILE, index=False)
        return raw_df
    return load_raw_survey()


def to_number(value: object) -> float:
    """Convert survey answers to floats while treating invalid option text as missing."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text or "option" in text.lower():
        return np.nan

    text = text.replace(",", "").replace("%", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def normalize_percentage(value: object, minimum_plausible: float = 30.0) -> float:
    """Normalize percentages entered as 0.70/70 and remove implausible values."""
    number = to_number(value)
    if pd.isna(number):
        return np.nan
    if 0 < number <= 1:
        number *= 100
    if number < minimum_plausible or number > 100:
        return np.nan
    return float(number)


def normalize_gpa(value: object) -> float:
    """Keep GPA values on the observed 0-4 scale and remove mixed-scale entries."""
    number = to_number(value)
    if pd.isna(number) or number < 0 or number > 4:
        return np.nan
    return float(number)


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns required for both regression experiments."""
    clean_df = raw_df.copy()
    clean_df.columns = clean_df.columns.str.strip()

    normalized_map = {key.strip(): value for key, value in COLUMN_MAP.items()}
    clean_df = clean_df.rename(columns=normalized_map)
    clean_df = clean_df.drop_duplicates()

    required_cols = ["CIA_Percentage", "GPA", "Attendance_Percentage"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns after renaming: {missing_cols}")

    clean_df["CIA_Percentage"] = clean_df["CIA_Percentage"].apply(normalize_percentage)
    clean_df["Attendance_Percentage"] = clean_df["Attendance_Percentage"].apply(
        normalize_percentage
    )
    clean_df["GPA"] = clean_df["GPA"].apply(normalize_gpa)

    return clean_df


def ols_parameters(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Calculate slope and intercept from the OLS closed-form equations."""
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)

    numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)
    if denominator == 0:
        raise ValueError("OLS slope is undefined because all X values are identical.")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def run_experiment(
    clean_df: pd.DataFrame,
    feature_col: str,
    target_col: str = "GPA",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Run scikit-learn and manual OLS regression for one feature."""
    data = clean_df[[feature_col, target_col]].dropna().copy()

    x = data[[feature_col]].to_numpy()
    y = data[target_col].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    sklearn_slope = float(model.coef_[0])
    sklearn_intercept = float(model.intercept_)
    sklearn_predictions = model.predict(x_test)

    manual_slope, manual_intercept = ols_parameters(x_train.ravel(), y_train)
    manual_predictions = manual_slope * x_test.ravel() + manual_intercept

    comparison = pd.DataFrame(
        {
            feature_col: x_test.ravel(),
            "Actual_GPA": y_test,
            "Sklearn_Prediction": sklearn_predictions,
            "Manual_OLS_Prediction": manual_predictions,
            "Absolute_Difference": np.abs(sklearn_predictions - manual_predictions),
        }
    )

    return {
        "feature": feature_col,
        "target": target_col,
        "rows_used": int(len(data)),
        "sklearn_slope": sklearn_slope,
        "sklearn_intercept": sklearn_intercept,
        "manual_slope": manual_slope,
        "manual_intercept": manual_intercept,
        "mse": float(mean_squared_error(y_test, sklearn_predictions)),
        "r2_score": float(r2_score(y_test, sklearn_predictions)),
        "comparison": comparison,
    }


def save_parameters(results: dict[str, dict[str, object]]) -> dict[str, object]:
    """Save learned slope/intercept values for both experiments."""
    parameters = {
        experiment_name: {
            "feature": result["feature"],
            "target": result["target"],
            "slope": result["sklearn_slope"],
            "intercept": result["sklearn_intercept"],
            "manual_ols_slope": result["manual_slope"],
            "manual_ols_intercept": result["manual_intercept"],
        }
        for experiment_name, result in results.items()
    }

    with PICKLE_FILE.open("wb") as file:
        pickle.dump(parameters, file)

    return parameters


def predict_from_loaded_parameters(
    loaded_parameters: dict[str, dict[str, object]],
    experiment_name: str,
    feature_value: float,
) -> float:
    """Predict GPA using slope and intercept loaded from pickle."""
    params = loaded_parameters[experiment_name]
    return float(params["slope"] * feature_value + params["intercept"])


def main() -> None:
    raw_df = export_csv_from_excel()
    clean_df = preprocess(raw_df)

    results = {
        "CIA_Percentage_to_GPA": run_experiment(clean_df, "CIA_Percentage"),
        "Attendance_Percentage_to_GPA": run_experiment(clean_df, "Attendance_Percentage"),
    }
    parameters = save_parameters(results)

    with PICKLE_FILE.open("rb") as file:
        loaded_parameters = pickle.load(file)

    print("Created:", CSV_FILE.name)
    print("Created:", PICKLE_FILE.name)
    print()

    for experiment_name, result in results.items():
        print(experiment_name)
        print("-" * len(experiment_name))
        print(f"Rows used: {result['rows_used']}")
        print(f"Scikit-learn slope: {result['sklearn_slope']:.10f}")
        print(f"Scikit-learn intercept: {result['sklearn_intercept']:.10f}")
        print(f"Manual OLS slope: {result['manual_slope']:.10f}")
        print(f"Manual OLS intercept: {result['manual_intercept']:.10f}")
        print(f"MSE: {result['mse']:.10f}")
        print(f"R2 score: {result['r2_score']:.10f}")
        print("Prediction comparison:")
        print(result["comparison"].round(6).to_string(index=False))
        print()

    print("Loaded pickle parameters:")
    print(parameters)
    print()
    print(
        "Example CIA=75 GPA prediction:",
        round(predict_from_loaded_parameters(loaded_parameters, "CIA_Percentage_to_GPA", 75), 4),
    )
    print(
        "Example Attendance=85 GPA prediction:",
        round(
            predict_from_loaded_parameters(
                loaded_parameters, "Attendance_Percentage_to_GPA", 85
            ),
            4,
        ),
    )


if __name__ == "__main__":
    main()
