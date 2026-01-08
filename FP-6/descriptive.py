import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def central(x):
    return np.mean(x), np.median(x), stats.mode(x, keepdims=True).mode[0]


def dispersion(x):
    return (
        np.std(x),
        np.min(x),
        np.max(x),
        np.max(x) - np.min(x),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.percentile(x, 75) - np.percentile(x, 25)
    )


def central_tendency_table(df: pd.DataFrame):
    numeric = df.select_dtypes(include=['float64', 'int64'])
    tbl = numeric.apply(lambda x: central(x), axis=0)
    tbl.index = ['mean', 'median', 'mode']
    print("\n=== CENTRAL TENDENCY ===")
    print(tbl)
    return tbl


def dispersion_table(df: pd.DataFrame):
    numeric = df.select_dtypes(include=['float64', 'int64'])
    tbl = numeric.apply(lambda x: dispersion(x), axis=0)
    tbl.index = ['st.dev', 'min', 'max', 'range', '25th', '75th', 'IQR']
    print("\n=== DISPERSION ===")
    print(tbl)
    return tbl


def corrcoeff(x, y):
    return np.corrcoef(x, y)[0, 1]


def plot_regression_line(ax, x, y, **kwargs):
    a, b = np.polyfit(x, y, 1)
    x0, x1 = min(x), max(x)
    ax.plot([x0, x1], [a*x0 + b, a*x1 + b], **kwargs)


def plot_descriptive(df):

    df_b = df[
        (df["pollutant"] == "PM2.5") &
        (df["category"] == "Total burden of disease") &
        (df["pollution_avg"] > 0) &
        (df["value"] > 0) &
        (df["value_per_100k"] > 0) &
        (df["population"] > 0)
    ].copy()

    if df_b.empty:
        print("No data after filtering for PM2.5 + total burden of disease.")
        return

    # Trim 99th percentile
    for col in ["pollution_avg", "value_per_100k", "value"]:
        upper = df_b[col].quantile(0.99)
        df_b = df_b[df_b[col] <= upper]

    # (a) DALYs per 100k vs pollution
    X_poll = df_b["pollution_avg"]
    Y_rate = df_b["value_per_100k"]

    # (b) Total DALYs vs population (in millions)
    X_pop_m = np.around(df_b["population"] / 1e6, 2)
    Y_total = df_b["value"]

    # (c) Age groups
    mask_children = df_b["age_group"].str.contains("< 15", na=False)
    mask_adults   = df_b["age_group"].str.contains(">= 25", na=False)

    # (d) Top-2 outcomes
    top2_outcomes = df_b["outcome"].value_counts().index[:2].tolist()

    sns.set_style("darkgrid")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

    # (a)
    ax = axs[0, 0]
    ax.scatter(X_poll, Y_rate, alpha=0.5, s=15)
    plot_regression_line(ax, X_poll, Y_rate, color="white", lw=2)
    ax.set_xlabel("PM2.5 pollution_avg [µg/m³]")
    ax.set_ylabel("DALYs per 100k (value_per_100k)")
    ax.set_title(f"(a) Burden vs pollution (all ages), r={corrcoeff(X_poll, Y_rate):.3f}")

    #  (b)
    ax = axs[0, 1]
    ax.scatter(X_pop_m, Y_total, alpha=0.5, s=15, color="tab:red")
    plot_regression_line(ax, X_pop_m, Y_total, color="white", lw=2)
    ax.set_xlabel("Population (millions)")
    ax.set_ylabel("Total DALYs (value)")
    ax.set_title(f"(b) Total burden vs population, r={corrcoeff(X_pop_m, Y_total):.3f}")

    #  (c)
    ax = axs[1, 0]
    for label, mask, col in [
        ("Children <15", mask_children, "tab:cyan"),
        ("Adults ≥25",   mask_adults,   "tab:purple")
    ]:
        d = df_b[mask]
        if d.empty:
            continue
        ax.scatter(d["pollution_avg"], d["value_per_100k"],
                   alpha=0.5, s=15, label=label, color=col)
        if len(d) >= 2:
            plot_regression_line(ax, d["pollution_avg"], d["value_per_100k"],
                                 color=col, lw=2)

    ax.set_xlabel("PM2.5 pollution_avg [µg/m³]")
    ax.set_ylabel("DALYs per 100k")
    ax.set_title("(c) Burden vs pollution by age group")
    ax.legend()

    # (d)
    ax = axs[1, 1]
    colors = ["tab:green", "tab:orange"]

    for outcome, col in zip(top2_outcomes, colors):
        d = df_b[df_b["outcome"] == outcome]
        if d.empty:
            continue

        ax.scatter(d["pollution_avg"], d["value_per_100k"],
                   alpha=0.5, s=15, label=outcome, color=col)

        if len(d) >= 2:
            plot_regression_line(ax, d["pollution_avg"], d["value_per_100k"],
                                 color=col, lw=2)
            r = corrcoeff(d["pollution_avg"], d["value_per_100k"])
            ax.text(
                0.03, 0.9 - 0.08 * colors.index(col),
                f"{outcome}: r={r:.3f}",
                transform=ax.transAxes,
                color=col,
            )

    ax.set_xlabel("PM2.5 pollution_avg [µg/m³]")
    ax.set_ylabel("DALYs per 100k")
    ax.set_title("(d) Burden vs pollution by outcome")
    ax.legend()

    plt.show()






