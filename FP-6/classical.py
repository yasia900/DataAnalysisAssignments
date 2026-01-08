from parse_data import get_clean_data
import pandas as pd
import numpy as np
from scipy import stats

df = get_clean_data("DataExtract.csv")

df = df[
    (df["pollutant"] == "PM2.5") &
    (df["category"] == "Total burden of disease") &
    (df["health_indicator"].str.contains("DALY", case=False, na=False)) &
    (df["pollution_avg"] > 0) &
    (df["value_per_100k"] > 0)
].copy()

df = df[df["pollution_avg"] <= df["pollution_avg"].quantile(0.99)]
df = df[df["value_per_100k"] <= df["value_per_100k"].quantile(0.99)]

print("Data shape after filtering:", df.shape)

# disease-specific PM2.5 sensitivity

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

# PM2.5

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

#Age × PM2.5 Interaction
print("\nTEST 3 — Age × PM2.5 interaction")

df["age_num"] = df["age_group"].str.extract(r"(\d+)").astype(float).fillna(0)

df["age_pm_interaction"] = df["age_num"] * df["pollution_avg"]

rho_int, p_int = stats.spearmanr(df["age_pm_interaction"], df["value_per_100k"])
print("Interaction Spearman rho =", rho_int)
print("p =", p_int)
