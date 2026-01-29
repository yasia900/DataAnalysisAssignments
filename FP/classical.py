import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from parse_data import get_clean_data


def prepare_analysis_data(filename="DataExtract.csv"):
    """Завантаження та фільтрація даних."""
    df = get_clean_data(filename)

    df = df[
        (df["pollutant"] == "PM2.5") &
        (df["category"] == "Total burden of disease") &
        (df["health_indicator"].str.contains("DALY", case=False, na=False)) &
        (df["pollution_avg"] > 0) &
        (df["value_per_100k"] > 0)
        ].copy()

    df = df[df["pollution_avg"] <= df["pollution_avg"].quantile(0.99)]
    df = df[df["value_per_100k"] <= df["value_per_100k"].quantile(0.99)]

    # Додаємо колонки для аналізу віку
    df["age_num"] = df["age_group"].str.extract(r"(\d+)").astype(float).fillna(0)
    df["age_class"] = pd.cut(
        df["age_num"],
        bins=[-1, 15, 50, 120],
        labels=["Children (<15)", "Adults (15–50)", "Older adults (50+)"]
    )
    df["age_pm_interaction"] = df["age_num"] * df["pollution_avg"]

    print("Data shape after filtering:", df.shape)
    return df


def run_statistical_tests(df):
    """Виконання статистичних тестів (ANOVA, Spearman)."""
    # TEST 1 — Disease-specific PM2.5 sensitivity
    print("\nTEST 1 — Disease-specific PM2.5 sensitivity")
    top_outcomes = df["outcome"].value_counts().index[:3]
    slopes_disease = {}

    for oc in top_outcomes:
        sub = df[df["outcome"] == oc]
        n = len(sub)
        if n < 30:
            print(f"{oc}: skipped (n={n} < 30)")
            continue
        slope, intercept = np.polyfit(sub["pollution_avg"], sub["value_per_100k"], 1)
        slopes_disease[oc] = (slope, n)

    for oc, (s, n) in slopes_disease.items():
        print(f"{oc}: n={n}, slope={s:.4f} (Δ DALY/100k per 1 µg/m³ PM2.5)")

    # TEST 2 — PM2.5
    print("\nTEST 2 — PM2.5")
    df["pm_decile"] = pd.qcut(df["pollution_avg"], 10, labels=False)
    groups = [df[df["pm_decile"] == i]["value_per_100k"] for i in range(10)]
    anova_dec = stats.f_oneway(*groups)
    print("ANOVA F =", anova_dec.statistic)
    print("ANOVA p =", anova_dec.pvalue)

    med_dec = df.groupby("pm_decile")["value_per_100k"].median()
    rho_dec, p_dec = stats.spearmanr(med_dec.index, med_dec.values)
    print("Spearman trend rho =", rho_dec)
    print("p =", p_dec)

    # TEST 3 — Age × PM2.5 interaction
    print("\nTEST 3 — Age × PM2.5 interaction")
    rho_int, p_int = stats.spearmanr(df["age_pm_interaction"], df["value_per_100k"])
    print("Interaction Spearman rho =", rho_int)
    print("p =", p_int)


def plot_environmental_impact(df):
    """Побудова графіків (Scatter та Bar)."""
    # Графік 1: Age × PM2.5 Interaction
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="pollution_avg", y="value_per_100k", hue="age_class", alpha=0.35)

    for group, sub in df.groupby("age_class", observed=False):
        if len(sub) > 2:
            sns.regplot(data=sub, x="pollution_avg", y="value_per_100k", scatter=False, label=f"{group} trend")

    plt.title("Figure 1: Age × PM2.5 interaction")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("DALY per 100k population")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Графік 2: Disease-specific sensitivity
    top3 = df["outcome"].value_counts().index[:3]
    slopes = []
    for oc in top3:
        sub = df[df["outcome"] == oc]
        slope, b = np.polyfit(sub["pollution_avg"], sub["value_per_100k"], 1)
        slopes.append(slope)

    slopes_df = pd.DataFrame({"Outcome": top3, "Slope": slopes})

    plt.figure(figsize=(8, 5))
    sns.barplot(data=slopes_df, x="Outcome", y="Slope", hue="Outcome", palette="Set2", legend=False)

    for i, v in enumerate(slopes_df["Slope"]):
        plt.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=10)

    plt.title("Figure 2: Disease-specific PM2.5 sensitivity")
    plt.ylabel("Slope (Δ DALY/100k per +1 µg/m³ PM2.5)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()