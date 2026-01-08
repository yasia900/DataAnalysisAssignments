import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


def prepare_ml_data(df):
    df_ml = df[
        (df["pollutant"] == "PM2.5") &
        (df["category"] == "Total burden of disease") &
        (df["value_per_100k"] > 0)
        ].copy()

    le_age = LabelEncoder()
    le_outcome = LabelEncoder()

    df_ml['age_encoded'] = le_age.fit_transform(df_ml['age_group'])
    df_ml['outcome_encoded'] = le_outcome.fit_transform(df_ml['outcome'])

    features = ['pollution_avg', 'age_encoded', 'outcome_encoded']
    X = df_ml[features]
    y = df_ml['value_per_100k']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_validated_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, cv_scores, y_pred


def plot_validation_results(y_test, y_pred):
    plt.figure(figsize=(8, 5))
    r2 = r2_score(y_test, y_pred)
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
    plt.title(f"Model Validation (R² = {r2:.3f})")
    plt.xlabel("Actual DALYs per 100k")
    plt.ylabel("Predicted DALYs per 100k")
    plt.show()


def plot_feature_importance(model):
    importances = model.feature_importances_
    features = ['Pollution Level', 'Age Group', 'Disease Type']
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, palette="magma")
    plt.title("Feature Importance: Drivers of Health Burden")
    plt.xlabel("Importance Score")
    plt.show()

def stability_analysis(df, n_runs=10):
    r2_scores = []

    for seed in range(n_runs):
        X_train, X_test, y_train, y_test = prepare_ml_data(df)
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=seed
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print("=== Stability Analysis: Random Forest ===")
    print(f"Number of runs: {n_runs}")
    print(f"Mean R²: {mean_r2:.3f}")
    print(f"Std R²: {std_r2:.3f}")
    print(f"All R² scores: {np.round(r2_scores, 3)}")

    return {
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "all_r2": r2_scores
    }

