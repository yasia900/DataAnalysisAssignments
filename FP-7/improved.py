import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def run_improvement_analysis(df):

    df_ml = df[(df["pollutant"] == "PM2.5") &
               (df["category"] == "Total burden of disease") &
               (df["value_per_100k"] > 0)].copy()

    le = LabelEncoder()
    df_ml['age_encoded'] = le.fit_transform(df_ml['age_group'])
    df_ml['outcome_encoded'] = le.fit_transform(df_ml['outcome'])

    features = ['pollution_avg', 'age_encoded', 'outcome_encoded']
    X = df_ml[features]
    y = df_ml['value_per_100k']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_base = RandomForestRegressor(n_estimators=100, random_state=42)
    model_base.fit(X_train, y_train)
    score_base = r2_score(y_test, model_base.predict(X_test))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model_imp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_imp.fit(X_train_s, y_train_s)
    score_imp = r2_score(y_test_s, model_imp.predict(X_test_s))

    return model_imp, X_test_s, y_test_s, score_base, score_imp, features


def plot_validation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Figure 1: Model Validation after Scaling (Improved)")
    plt.xlabel("Actual DALYs")
    plt.ylabel("Predicted DALYs")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_importance(model, feature_names):
    importances = model.feature_importances_
    # Гарні назви для графіку
    clean_labels = [n.replace('_', ' ').title() for n in feature_names]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=clean_labels, palette="viridis")
    plt.title("Figure 2: Feature Importance after Scaling (Improved)")
    plt.xlabel("Importance Weight")
    plt.show()


def plot_comparison(base_score, imp_score):
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Baseline (FP-6)', 'Improved (Scaling)'], [base_score, imp_score], color=['#A9A9A9', '#2E8B57'])
    plt.title("Figure 3: R² Score Comparison (Baseline vs. Scaled)")
    plt.ylabel("$R^2$ Score")
    plt.ylim(0, 1.0)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{bar.get_height():.4f}', ha='center',
                 fontweight='bold')
    plt.show()