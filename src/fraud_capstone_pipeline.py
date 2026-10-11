# =============================================================================
# AI-Powered Fraud Detection and Risk Scoring System for Agribank Online Mob App
#
# V2 changes
#   1. Calibration and the cost based threshold now picked on VALIDATION.
#      v1: I picked them on the test set. That leaks the test set into a decision, so the test score stops being a clean check.
#   2. All four models are now tuned and compared on Feature Set D (E6). On Set A every model scores about 1.000, so Set A cannot rank them.
#   3. The E6 model is tuned on its own features. In v1 it reused the settings that were tuned on Set A.
#   4. The main alert rule is capacity-based (top k per day). Fixed thresholds are still reported, but only for comparison.
#   5. The bias review now uses the same alert rule as the report. In v1 the code and the report used different thresholds.
#   6. New mitigation test: drop the time-of-day features and compare.
#
# =============================================================================

# %% --------------------------------------------------------------------------
# SECTION 0. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------
import os
import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Remember: remove this line if I want plots to pop up in Jupyter.
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import ParameterSampler, train_test_split, StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss, precision_recall_curve,
    roc_curve, accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)   # this to show every column in printed tables

CONFIG = {
    # Paths
    "data_path": os.environ.get("PAYSIM_PATH", "data/raw/PS_20174392719_1491204439457_log.csv"),
    "models_dir": "models",
    "figures_dir": "reports/figures",
    "tables_dir": "reports/tables",

    # Reproducibility
    "random_state": 42,

    # Time-based split (share of rows, in time order) - PART VIII, Section A
    "train_share": 0.70,
    "val_share": 0.15,

    # Training data reduction. Legitimate rows in the TRAINING set only are
    # sampled down to this share to keep training time reasonable.
    # Remember: val and test are never touched by sampling or rebalancing, keep these clean.
    "legit_train_fraction": 0.10,

    # Optional SMOTE on the training set only - PART VIII, Section B
    "use_smote": False,

    # Tuning effort per model (number of random settings tried)
    "n_search_iter": 8,
    # CHANGE (v2): Set A is already near perfect, so more tuning there changes
    # nothing. Set D (E6) is where the models differ, so it gets more tries.
    "n_search_iter_realistic": 20,

    # Operational assumptions (illustrative only, not Agribank figures) - PART IX
    "alerts_per_day_capacity": 50,
    "capacity_options": [50, 150, 300],        # alerts per day to compare (staffing view)
    # CHANGE (v2): the main alert rule for the report. The team reviews the
    # top 150 scores per day. 150 is my assumption, not an Agribank figure.
    "operating_capacity": 150,
    # Fixed thresholds shown next to the capacity rule, for comparison only.
    "compare_thresholds": [0.05, 0.01],
    # CHANGE (v2): bootstrap rounds for a 95% interval on test PR-AUC.
    # 200 is enough for a rough interval and still runs in a few minutes.
    "n_bootstrap": 200,
    "review_costs": [100, 500, 2000],          # assumed cost per alert, simulated currency units
    "thresholds": [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],

    # Model selection: models within this PR-AUC margin of the best are treated
    # as tied, and the first one in this preference list is chosen.
    # NOTE TO SELF: tree models come first because SHAP explains them directly.
    "selection_tolerance": 0.002,
    "selection_preference": ["XGBoost", "RandomForest", "DecisionTree", "LogisticRegression"],

    # Explainability sample size - PART X
    "shap_sample": 5000,

    # Quick dry run: set to a number (e.g. 500_000) to load only the first N rows.
    # To do: return this back to None before pushing the final run.
    "nrows": None,

    # CHANGE (v2): Word report used by Section 17. Put my report in reports/
    # and the script writes a filled copy next to it.
    "report_template": "reports/FJVP-AIML-Capstone_Technical_Report_Sheet.docx",
    "report_output": "reports/FJVP-AIML-Capstone_Technical_Report_Final.docx",
}

for d in [CONFIG["models_dir"], CONFIG["figures_dir"], CONFIG["tables_dir"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

RS = CONFIG["random_state"]
np.random.seed(RS)


def save_table(df, name):
    """Save a table for the report or decks and return it for display."""
    path = Path(CONFIG["tables_dir"]) / f"{name}.csv"
    df.to_csv(path, index=True)
    print(f"  saved table: {path}")
    return df


def save_fig(name):
    path = Path(CONFIG["figures_dir"]) / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved figure: {path}")


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# CHANGE (v2): one place to collect every number the report quotes.
# Each key matches a {{KEY}} placeholder in the Word report.
# In v1 I typed numbers into the report by hand, and some went out of date.
REPORT = {}


def rep(key, value):
    REPORT[key] = value
    return value


def f_int(x):
    return f"{int(round(float(x))):,}"


def f_pct(x, d=1):
    return f"{float(x) * 100:.{d}f}%"


def f4(x):
    return f"{float(x):.4f}"


def f3(x):
    return f"{float(x):.3f}"


def f_money(x):
    return f"{float(x):,.2f}"


NICE = {"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree",
        "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
ABBR = {"LogisticRegression": "LR", "DecisionTree": "DT", "RandomForest": "RF", "XGBoost": "XGB"}
PARAM_LABEL = {"C": "C", "max_depth": "max depth", "min_samples_leaf": "min samples per leaf",
               "n_estimators": "trees", "max_features": "features per split", "learning_rate": "learning rate",
               "subsample": "row sample", "colsample_bytree": "column sample", "min_child_weight": "min child weight"}


def fmt_params(params):
    """Short readable version of a settings dict for the report tables."""
    parts = []
    for k, v in params.items():
        v = "no limit" if v is None else v
        parts.append(f"{PARAM_LABEL.get(k, k)} {v}")
    return ", ".join(parts)


# %% ---------------------------------------------------------------------------
# SECTION 1. PROBLEM FRAMING                                  [Instr. Step 1]
# Report: PART III (Problem Understanding) and PART IV (Success Metrics)
# -----------------------------------------------------------------------------
section("SECTION 1. PROBLEM FRAMING")

PROBLEM = {
    "business_problem": (
        "How can machine learning help identify potentially fraudulent Agribank Online "
        "transactions, and assign a risk score that helps fraud personnel decide which "
        "transactions to review first?"
    ),
    "task_type": "Supervised binary classification (isFraud: 0 = legitimate, 1 = fraud)",
    "why_classification": (
        "The label is known for past transactions and the outcome has two classes. "
        "The model's probability output becomes the 0 to 100 risk score."
    ),
    "scoring_point": "Before the transaction is completed. Only information available at that moment may be used.",
    "main_metric": "PR-AUC (average precision), because fraud is about 0.13% of records",
    "supporting_metrics": ["Precision", "Recall", "F1", "ROC-AUC", "False positive rate",
                           "Precision@k / Recall@k", "Brier score"],
    "business_kpis": ["Fraud detection rate", "Fraud value detected", "Alert volume",
                      "False alert volume", "Investigation efficiency", "Potential fraud loss exposure"],
}
for k, v in PROBLEM.items():
    print(f"- {k}: {v}")

# Leaving this text block here: script spells out reasoning for code reviewers looking for Step 1.


# %% ---------------------------------------------------------------------------
# SECTION 2. DATA LOADING AND UNDERSTANDING                   [Inst. Step 2]
# Report: PART V (Data Collection and Understanding)
# -----------------------------------------------------------------------------
section("SECTION 2. DATA LOADING AND UNDERSTANDING")

# Source: Kaggle, "Synthetic Financial Datasets for Fraud Detection" (PaySim),
# file PS_20174392719_1491204439457_log.csv
# Don't forget to drop the Kaggle link & Lopez-Rojas paper into the README later.

DTYPES = {
    "step": "int16", "type": "category", "amount": "float64",
    "nameOrig": "string", "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
    "nameDest": "string", "oldbalanceDest": "float64", "newbalanceDest": "float64",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}

t0 = time.time()
df = pd.read_csv(CONFIG["data_path"], dtype=DTYPES, nrows=CONFIG["nrows"])
print(f"Loaded {len(df):,} rows x {df.shape[1]} columns in {time.time() - t0:.1f}s")

# 2.1 Overview: types, missing values, duplicates
overview = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(4),
    "unique_values": df.nunique(),
})
save_table(overview, "02_dataset_overview")
print(overview)
print(f"Duplicate rows: {df.duplicated().sum():,}")

# 2.2 Data dictionary with ranges and units - PART V, Section C
ROLES = {
    "step": ("Simulated hour (1 = first hour)", "Input", "Yes (timestamp)"),
    "type": ("Transaction type", "Input", "Yes"),
    "amount": ("Transaction amount", "Input", "Yes"),
    "nameOrig": ("Sender account", "Identifier", "Yes, history only"),
    "oldbalanceOrg": ("Sender balance before", "Input", "Yes"),
    "newbalanceOrig": ("Sender balance after", "Input (derived)", "Yes, old balance less amount"),
    "nameDest": ("Receiving account", "Identifier", "Yes, history only"),
    "oldbalanceDest": ("Receiving balance before", "Input", "On-us only"),
    "newbalanceDest": ("Receiving balance after", "Input", "No for interbank"),
    "isFraud": ("Fraud label", "Target", "No (known after investigation)"),
    "isFlaggedFraud": ("Simulator rule output", "Benchmark only", "Not an input"),
}
rows = []
for col in df.columns:
    desc, role, avail = ROLES.get(col, ("", "", ""))
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        rng = f"{s.min():,.2f} to {s.max():,.2f}"
    else:
        rng = f"{s.nunique():,} distinct values"
    unit = "simulated currency" if "balance" in col or col == "amount" else ("hours" if col == "step" else "-")
    rows.append([col, str(s.dtype), desc, rng, unit, role, avail])
data_dict = pd.DataFrame(rows, columns=["column", "dtype", "description", "range", "unit", "role",
                                        "available_at_scoring"]).set_index("column")
save_table(data_dict, "02_data_dictionary")

# CHANGE (v2): the report's data dictionary now shows ranges and units.
# Step 2 asks for "ranges/units" and v1 only had them in this CSV.
MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
for col in MONEY_COLS:
    rep(f"RANGE_{col}", f"{df[col].min():,.2f} to {df[col].max():,.2f}")

# CHANGE (v2): distribution summary for PART V. The medians, skew and share
# of zeros show how lopsided the money fields are. That is why I use logs.
dist = pd.DataFrame({
    "median": df[MONEY_COLS].median(),
    "mean": df[MONEY_COLS].mean(),
    "skew": df[MONEY_COLS].skew(),
    "zero_share": (df[MONEY_COLS] == 0).mean(),
})
save_table(dist.round(4), "02_distribution_summary")
for col in MONEY_COLS:
    rep(f"DS_{col}_median", f_money(dist.loc[col, "median"]))
    rep(f"DS_{col}_mean", f_money(dist.loc[col, "mean"]))
    rep(f"DS_{col}_skew", f"{dist.loc[col, 'skew']:.1f}")
    rep(f"DS_{col}_zero", f_pct(dist.loc[col, "zero_share"]))

# 2.3 Class balance
class_counts = df["isFraud"].value_counts().rename({0: "legitimate", 1: "fraud"})
print(class_counts)
print(f"Fraud share: {df['isFraud'].mean() * 100:.4f}%  (about 1 in {1 / df['isFraud'].mean():,.0f})")

# 2.4 Verify the known PaySim characteristics - PART V, Section D
# Treating PaySim's readme claims as hypotheses here, not truth. check/ see what the data actually says ^^.
checks = {}
fraud = df[df["isFraud"] == 1]
checks["fraud_types"] = sorted(fraud["type"].astype(str).unique().tolist())
checks["share_fraud_emptying_sender"] = float(np.isclose(fraud["amount"], fraud["oldbalanceOrg"]).mean())
checks["share_fraud_dest_zero_before_after"] = float(((fraud["oldbalanceDest"] == 0) & (fraud["newbalanceDest"] == 0)).mean())
checks["share_legit_dest_zero_before_after"] = float(((df.loc[df.isFraud == 0, "oldbalanceDest"] == 0) &
                                                      (df.loc[df.isFraud == 0, "newbalanceDest"] == 0)).mean())
checks["sender_accounts_unique_share"] = float(df["nameOrig"].nunique() / len(df))
checks["merchant_dest_share"] = float(df["nameDest"].str.startswith("M").mean())
checks["flagged_count"] = int(df["isFlaggedFraud"].sum())
checks["flagged_min_amount"] = float(df.loc[df.isFlaggedFraud == 1, "amount"].min()) if checks["flagged_count"] else None
checks["flagged_types"] = sorted(df.loc[df.isFlaggedFraud == 1, "type"].astype(str).unique().tolist())
checks["max_step"] = int(df["step"].max())
for k, v in checks.items():
    print(f"- {k}: {v}")
with open(Path(CONFIG["tables_dir"]) / "02_known_characteristics.json", "w") as f:
    json.dump(checks, f, indent=2)


# %% ---------------------------------------------------------------------------
# SECTION 3. EXPLORATORY DATA ANALYSIS                        [Inst. Step 3]
# Report: PART VI, Section B
# -----------------------------------------------------------------------------
section("SECTION 3. EXPLORATORY DATA ANALYSIS")

# 3.1 Fraud by transaction type
by_type = df.groupby("type", observed=True)["isFraud"].agg(transactions="size", fraud="sum")
by_type["fraud_rate_pct"] = (by_type["fraud"] / by_type["transactions"] * 100).round(4)
save_table(by_type, "03_fraud_by_type")
ax = by_type["transactions"].plot(kind="bar", color="#3A7C22", figsize=(7, 4), title="Transactions by type")
ax.set_ylabel("Count")
save_fig("03_transactions_by_type")

# 3.2 Amount distribution, fraud vs legitimate (log scale)
plt.figure(figsize=(7, 4))
for label, color in [(0, "#3A7C22"), (1, "#9B1D1D")]:
    vals = np.log10(df.loc[df.isFraud == label, "amount"].clip(lower=1))
    plt.hist(vals, bins=60, alpha=0.6, density=True, color=color, label="Fraud" if label else "Legitimate")
plt.xlabel("log10(amount)"); plt.ylabel("Density"); plt.legend(); plt.title("Amount distribution")
save_fig("03_amount_distribution")

# 3.3 Outliers in amount (IQR rule) - reported, not removed
# NOTE TO SELF: leaving outliers alone—fraud lives in the tail, & dropping them means throwing away the actual cases need 2 catch.
q1, q3 = df["amount"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
outlier_summary = pd.DataFrame({
    "iqr_upper_bound": [upper],
    "outlier_rows": [(df["amount"] > upper).sum()],
    "outlier_share_pct": [(df["amount"] > upper).mean() * 100],
    "fraud_rate_in_outliers_pct": [df.loc[df["amount"] > upper, "isFraud"].mean() * 100],
    "fraud_rate_overall_pct": [df["isFraud"].mean() * 100],
})
save_table(outlier_summary, "03_outlier_summary")
rep("OUT_upper", f_money(upper))
rep("OUT_rows", f_int(outlier_summary["outlier_rows"].iloc[0]))
rep("OUT_share", f"{outlier_summary['outlier_share_pct'].iloc[0]:.1f}%")
rep("OUT_fraud_rate", f"{outlier_summary['fraud_rate_in_outliers_pct'].iloc[0]:.2f}%")

# CHANGE (v2): median amount, fraud vs legitimate, inside TRANSFER and
# CASH_OUT only. Comparing against all types would be unfair because
# PAYMENT amounts are small and never fraud.
in_scope = df["type"].isin(["TRANSFER", "CASH_OUT"])
rep("EDA_FRAUD_MED_AMT", f_money(df.loc[in_scope & (df.isFraud == 1), "amount"].median()))
rep("EDA_LEGIT_MED_AMT", f_money(df.loc[in_scope & (df.isFraud == 0), "amount"].median()))

# CHANGE (v2): share of fraud TRANSFERs that have a CASH_OUT fraud with the
# same amount in the same hour. This backs up the "fraud pairs" sentence.
f_tr = df[(df.isFraud == 1) & (df["type"] == "TRANSFER")][["step", "amount"]]
f_co = df[(df.isFraud == 1) & (df["type"] == "CASH_OUT")][["step", "amount"]].drop_duplicates()
paired = f_tr.merge(f_co, on=["step", "amount"], how="left", indicator=True)["_merge"].eq("both").mean()
rep("EDA_PAIR_SHARE", f_pct(paired))

# 3.4 Fraud by hour of day and by day
df["hour"] = (df["step"] % 24).astype("int8")
df["day"] = (df["step"] // 24).astype("int16")
hourly = df.groupby("hour")["isFraud"].agg(transactions="size", fraud="sum")
hourly["fraud_rate_pct"] = hourly["fraud"] / hourly["transactions"] * 100
save_table(hourly, "03_fraud_by_hour")
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(hourly.index, hourly["transactions"], color="#C9D9C0", label="Transactions")
ax2 = ax1.twinx()
ax2.plot(hourly.index, hourly["fraud_rate_pct"], color="#9B1D1D", marker="o", label="Fraud rate %")
ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Transactions"); ax2.set_ylabel("Fraud rate (%)")
plt.title("Volume and fraud rate by hour")
save_fig("03_fraud_by_hour")

daily = df.groupby("day")["isFraud"].agg(transactions="size", fraud="sum")
save_table(daily, "03_fraud_by_day")

# CHANGE (v2): night (00:00 to 05:59) vs the rest of the day. The bias
# review keeps coming back to time of day, so I want the raw rates too.
night = df["hour"].between(0, 5)
df_night_rate, df_day_rate = df.loc[night, "isFraud"].mean(), df.loc[~night, "isFraud"].mean()
rep("EDA_NIGHT_RATE", f"{df_night_rate * 100:.2f}%")
rep("EDA_DAY_RATE", f"{df_day_rate * 100:.2f}%")

# CHANGE (v2): daily legitimate vs fraud volume. This is the chart that
# explains why the test period has a higher fraud rate than training.
daily["legit"] = daily["transactions"] - daily["fraud"]
half = daily.index.max() // 2
rep("EDA_LEGIT_DAY_FIRST", f_int(daily.loc[daily.index <= half, "legit"].mean()))
rep("EDA_LEGIT_DAY_LAST", f_int(daily.loc[daily.index > half, "legit"].mean()))
rep("EDA_FRAUD_DAY_FIRST", f_int(daily.loc[daily.index <= half, "fraud"].mean()))
rep("EDA_FRAUD_DAY_LAST", f_int(daily.loc[daily.index > half, "fraud"].mean()))
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(daily.index, daily["legit"], color="#3A7C22", marker="o", label="Legitimate")
ax1.set_xlabel("Simulated day"); ax1.set_ylabel("Legitimate transactions")
ax2 = ax1.twinx()
ax2.plot(daily.index, daily["fraud"], color="#9B1D1D", marker="s", label="Fraud")
ax2.set_ylabel("Fraud transactions")
plt.title("Daily volume: legitimate vs fraud")
save_fig("03_daily_volume")
# NOTE Lang: # PaySim shouldn't have nulls, but just in case, shd fill missing balances with 0 and drop untagged rows.

# 3.5 Correlation of numeric fields
num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
corr = df[num_cols].corr()
save_table(corr.round(3), "03_correlation_matrix")
plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="Greens", vmin=-1, vmax=1)
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right"); plt.yticks(range(len(num_cols)), num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(); plt.title("Correlation matrix")
save_fig("03_correlation_matrix")


# %% ---------------------------------------------------------------------------
# SECTION 4. PREPROCESSING AND FEATURE ENGINEERING            [Instr. Step 3]
# Report: PART VI, Sections A, C and D
# -----------------------------------------------------------------------------
section("SECTION 4. PREPROCESSING AND FEATURE ENGINEERING")

# 4.1 Missing values and duplicates
# NOTE TO SELF: PaySim is expected to have no nulls. If any appear, the rule
# below fills balances with 0 and drops rows missing the label. Write the
# actual counts in the report either way.
before = len(df)
df = df.dropna(subset=["isFraud", "type", "amount"])
bal_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
df[bal_cols] = df[bal_cols].fillna(0)
df = df.drop_duplicates()
print(f"Rows removed by null/duplicate handling: {before - len(df):,}")

# 4.2 Receiving account history (computed on ALL transaction types, past steps only)
# For each (receiving account, step) we count what that account received in
# EARLIER steps. Transactions in the same step are not counted.
agg = df.groupby(["nameDest", "step"], observed=True)["amount"].agg(n="size", s="sum")
cum = agg.groupby(level=0).cumsum()
prior = (cum - agg).rename(columns={"n": "dest_prior_count", "s": "dest_prior_amount"})
df = df.join(prior, on=["nameDest", "step"])
df["dest_is_new"] = (df["dest_prior_count"] == 0).astype("int8")
del agg, cum, prior

# 4.3 Scope filter - PART VI, Section A
fraud_types = set(checks["fraud_types"])
if fraud_types.issubset({"TRANSFER", "CASH_OUT"}):
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df["type"] = df["type"].cat.remove_unused_categories()
    print(f"Scope filter applied: TRANSFER and CASH_OUT only, {len(df):,} rows remain")
else:
    print(f"Scope filter NOT applied: fraud found in {fraud_types}")
# NOTES: this scope filter acts as a pre-filter rule before hitting the model in a real deployment.

# 4.4 Features that do not depend on training statistics
df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["orig_emptied"] = np.isclose(df["amount"], df["oldbalanceOrg"]).astype("int8")
df["dest_zero_after"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)).astype("int8")
df["log_amount"] = np.log1p(df["amount"])
df["amt_to_orig_bal"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["amt_to_dest_bal"] = df["amount"] / (df["oldbalanceDest"] + 1)
df["log_dest_prior_amount"] = np.log1p(df["dest_prior_amount"])
df["period_night"] = df["hour"].between(0, 5).astype("int8")
df["period_morning"] = df["hour"].between(6, 11).astype("int8")
df["period_afternoon"] = df["hour"].between(12, 17).astype("int8")
# evening is the remaining group, left out to avoid a redundant column
df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype("int8")

# 4.5 Time-based split - PART VIII, Section A
df = df.sort_values("step").reset_index(drop=True)
n = len(df)
cut_train = df["step"].iloc[int(n * CONFIG["train_share"])]
cut_val = df["step"].iloc[int(n * (CONFIG["train_share"] + CONFIG["val_share"]))]
train = df[df["step"] < cut_train].copy()
val = df[(df["step"] >= cut_train) & (df["step"] < cut_val)].copy()
test = df[df["step"] >= cut_val].copy()
split_summary = pd.DataFrame({
    "steps": [f"1 to {cut_train - 1}", f"{cut_train} to {cut_val - 1}", f"{cut_val} to {df.step.max()}"],
    "rows": [len(train), len(val), len(test)],
    "fraud": [train.isFraud.sum(), val.isFraud.sum(), test.isFraud.sum()],
}, index=["train", "validation", "test"])
split_summary["fraud_rate_pct"] = (split_summary["fraud"] / split_summary["rows"] * 100).round(4)
save_table(split_summary, "04_time_split_summary")
print(split_summary)
# NOTE: make sure lang each split holds enough fraud cases; tweak train/val shares if things look too thin.

# 4.6 Features that use training statistics (fitted on TRAIN only, then applied)
high_value_cut = train["amount"].quantile(0.99)
type_median = train.groupby("type", observed=True)["amount"].median()
for part in (train, val, test):
    part["high_value"] = (part["amount"] >= high_value_cut).astype("int8")
    med = part["type"].map(type_median).astype(float)
    part["amt_vs_type_median"] = np.log1p(part["amount"]) - np.log1p(med)
REFERENCE_STATS = {"high_value_cut": float(high_value_cut),
                   "type_median": {str(k): float(v) for k, v in type_median.items()}}

# 4.7 Feature sets - PART VI, Section D and PART VIII, Section F
BASE = ["log_amount", "amt_to_orig_bal", "high_value", "amt_vs_type_median",
        "oldbalanceOrg", "newbalanceOrig", "errorBalanceOrig", "orig_emptied",
        "hour", "period_night", "period_morning", "period_afternoon", "type_TRANSFER",
        "dest_prior_count", "log_dest_prior_amount", "dest_is_new"]
RECEIVING_BAL = ["oldbalanceDest", "newbalanceDest", "errorBalanceDest", "dest_zero_after", "amt_to_dest_bal"]
RECON = ["errorBalanceOrig", "errorBalanceDest", "orig_emptied", "dest_zero_after"]

FEATURE_SETS = {
    "A_full": BASE + RECEIVING_BAL,                                   # on-us view, headline
    "A_no_recon": [f for f in BASE + RECEIVING_BAL if f not in RECON],  # stress test E2
    "B_interbank": BASE,                                              # no receiving balances
    "A_strict": [f for f in BASE + RECEIVING_BAL                      # stress test E5
                 if f not in RECON + ["newbalanceOrig", "amt_to_orig_bal", "oldbalanceOrg"]],
    "T_transaction_only": [f for f in BASE                            # stress test E6 (realistic view)
                           if f not in RECON + ["newbalanceOrig", "amt_to_orig_bal", "oldbalanceOrg"]],
}
# NOTE TO SELF: A_strict also removes the sender balance fields, because E2
# can still partly detect an emptied account from them.
# T_transaction_only (E6) goes further and drops ALL balance fields, sender and
# receiving. It keeps only what Agribank would know when a customer submits a
# transfer: amount, type, time and receiving account history. This is the
# realistic view used for the business deck.
# NOTE TO SELF: newbalanceDest is only known after the transfer. It sits in
# Feature Set A because the report treats Set A as the "full PaySim" headline
# view. Set B is the interbank view (no receiving balances).
# CHANGE (v2): in v1 this note called Set B "the realistic view". That was
# wrong. Set D (T_transaction_only, test E6) is the realistic view, because
# it is the only set with no balance fields at all. The report says Set D.

TARGET = "isFraud"
for name, cols in FEATURE_SETS.items():
    print(f"{name}: {len(cols)} features")


# %% ---------------------------------------------------------------------------
# SECTION 5. TRAINING DATA PREPARATION (imbalance)            [Inst. Step 3/4}
# Report: PART V, Section F and PART VIII, Section B
# -----------------------------------------------------------------------------
section("SECTION 5. TRAINING DATA PREPARATION")


def make_training_sample(train_df, frac, seed=RS):
    """Keep all fraud rows and a fraction of legitimate rows. Training set only."""
    fraud_rows = train_df[train_df[TARGET] == 1]
    legit_rows = train_df[train_df[TARGET] == 0].sample(frac=frac, random_state=seed)
    return pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=seed)


train_s = make_training_sample(train, CONFIG["legit_train_fraction"])
print(f"Training sample: {len(train_s):,} rows, fraud share {train_s[TARGET].mean() * 100:.2f}%")

if CONFIG["use_smote"]:
    from imblearn.over_sampling import SMOTE
    # NOTE TO SELF: SMOTE runs on the training sample only, never on val/test.
    print("SMOTE enabled for training data")


def fit_xy(cols, sample=train_s):
    X, y = sample[cols], sample[TARGET]
    if CONFIG["use_smote"]:
        X, y = SMOTE(random_state=RS).fit_resample(X, y)
    return X, y


# %% ---------------------------------------------------------------------------
# SECTION 6. FEATURE SELECTION AND PCA                        [Inst. Step 3]
# Report: PART VII
# -----------------------------------------------------------------------------
section("SECTION 6. FEATURE SELECTION AND PCA")

COLS_A = FEATURE_SETS["A_full"]

# 6.1 Mutual information (filter method)
mi_sample = train_s.sample(min(200_000, len(train_s)), random_state=RS)
mi = mutual_info_classif(mi_sample[COLS_A], mi_sample[TARGET], random_state=RS)
mi_table = pd.Series(mi, index=COLS_A, name="mutual_information").sort_values(ascending=False).to_frame()

# 6.2 Model-based importance (Random Forest) + permutation check on validation
rf_sel = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                n_jobs=-1, random_state=RS)
rf_sel.fit(train_s[COLS_A], train_s[TARGET])
mi_table["rf_importance"] = pd.Series(rf_sel.feature_importances_, index=COLS_A)
val_sample = val.sample(min(100_000, len(val)), random_state=RS)
perm = permutation_importance(rf_sel, val_sample[COLS_A], val_sample[TARGET], scoring="average_precision",
                              n_repeats=3, random_state=RS, n_jobs=-1)
mi_table["permutation_importance"] = pd.Series(perm.importances_mean, index=COLS_A)
save_table(mi_table.round(5), "06_feature_selection")
print(mi_table.round(4))

# 6.3 Selection rule
# Drop a feature only if it is weak on ALL three measures. Highly correlated
# pairs are reported so I can decide manually.
weak = mi_table[(mi_table["mutual_information"] < 0.001) &
                (mi_table["rf_importance"] < 0.005) &
                (mi_table["permutation_importance"] <= 0)].index.tolist()
print(f"Weak on all three measures (candidates to drop): {weak}")
corr_a = train_s[COLS_A].corr().abs()
pairs = [(a, b, round(corr_a.loc[a, b], 3)) for i, a in enumerate(COLS_A) for b in COLS_A[i + 1:]
         if corr_a.loc[a, b] > 0.95]
print(f"Highly correlated pairs (> 0.95): {pairs}")
rep("FS_WEAK", ", ".join(weak) if weak else "None")
rep("FS_TOP_MI", mi_table["mutual_information"].idxmax())
rep("FS_TOP_RF", mi_table["rf_importance"].idxmax())
rep("FS_TOP_PERM", mi_table["permutation_importance"].idxmax())
for key in FEATURE_SETS:
    FEATURE_SETS[key] = [f for f in FEATURE_SETS[key] if f not in weak]
COLS_A = FEATURE_SETS["A_full"]
# Make sure to justify *why* we dropped these in the writeup, not just list them.

# 6.4 PCA (fitted on training data only)
scaler_pca = StandardScaler().fit(train_s[COLS_A])
pca = PCA(random_state=RS).fit(scaler_pca.transform(train_s[COLS_A]))
evr = pd.DataFrame({"explained_variance": pca.explained_variance_ratio_,
                    "cumulative": np.cumsum(pca.explained_variance_ratio_)},
                   index=[f"PC{i + 1}" for i in range(len(COLS_A))])
save_table(evr.round(4), "06_pca_explained_variance")
n95 = int(np.argmax(evr["cumulative"].values >= 0.95) + 1)
print(f"Components needed for 95% of variance: {n95} of {len(COLS_A)}")

plt.figure(figsize=(7, 4))
plt.bar(range(1, len(evr) + 1), evr["explained_variance"], color="#3A7C22")
plt.plot(range(1, len(evr) + 1), evr["cumulative"], color="#0C3512", marker="o")
plt.axhline(0.95, ls="--", color="grey"); plt.xlabel("Component"); plt.ylabel("Share of variance")
plt.title("PCA explained variance")
save_fig("06_pca_explained_variance")

pts = pca.transform(scaler_pca.transform(mi_sample[COLS_A]))[:, :2]
plt.figure(figsize=(6, 5))
lg = mi_sample[TARGET].values == 0
plt.scatter(pts[lg, 0], pts[lg, 1], s=2, alpha=0.2, color="#3A7C22", label="Legitimate")
plt.scatter(pts[~lg, 0], pts[~lg, 1], s=4, alpha=0.6, color="#9B1D1D", label="Fraud")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.title("Training sample on first two components")
save_fig("06_pca_2d")

# Does PCA help Logistic Regression?
lr_plain = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
lr_pca = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=n95, random_state=RS)),
                   ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
pca_test = {}
for name, m in [("LR without PCA", lr_plain), ("LR with PCA", lr_pca)]:
    m.fit(train_s[COLS_A], train_s[TARGET])
    pca_test[name] = average_precision_score(val[TARGET], m.predict_proba(val[COLS_A])[:, 1])
save_table(pd.Series(pca_test, name="val_pr_auc").to_frame(), "06_pca_vs_no_pca")
print(pca_test)
rep("PCA_N95", str(n95))
rep("PCA_NFEAT", str(len(COLS_A)))
rep("PCA_LR_PLAIN", f4(pca_test["LR without PCA"]))
rep("PCA_LR_PCA", f4(pca_test["LR with PCA"]))
# CHANGE (v2): lr_plain uses the default C = 1. The tuned Logistic Regression
# in Section 8 uses C = 0.01. That is why this check and the tuning table can
# show slightly different validation PR-AUC. Both are right. The report now
# says "default settings" here so a reader does not think it is a typo.
# NOTE: if PCA doesn't give a clear edge, ditch it. Plain features are way easier to explain to risk analysts anyway.


# %% ---------------------------------------------------------------------------
# SECTION 7. EVALUATION HELPERS                                [Inst. Step 4]
# Report: PART IV
# -----------------------------------------------------------------------------
def precision_recall_at_k(y, p, step, k_per_day):
    """Top-k transactions per simulated day, k = review capacity."""
    tmp = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "day": np.asarray(step) // 24})
    tmp["rank"] = tmp.groupby("day")["p"].rank(ascending=False, method="first")
    picked = tmp["rank"] <= k_per_day
    tp = tmp.loc[picked, "y"].sum()
    return tp / max(picked.sum(), 1), tp / max(tmp["y"].sum(), 1), int(picked.sum())


def capacity_alerts(p, step, k_per_day):
    """CHANGE (v2): turn scores into alerts with the capacity rule.
    The top k scores in each simulated day become alerts. No threshold needed,
    so a shift in the fraud rate cannot blow up the alert volume."""
    tmp = pd.DataFrame({"p": np.asarray(p), "day": np.asarray(step) // 24})
    rank = tmp.groupby("day")["p"].rank(ascending=False, method="first")
    return (rank <= k_per_day).astype(int).values


def bootstrap_pr_auc(y, p, n_boot, seed=RS):
    """CHANGE (v2): 95% bootstrap interval for test PR-AUC.
    Resample test rows with replacement and recompute PR-AUC each time.
    If two models' intervals overlap a lot, I should not call one "better"."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() > 0:
            scores.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(scores, [2.5, 97.5])


def evaluate(y, p, frame, threshold=0.5, label=""):
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    amt = frame["amount"].values
    p_at_k, r_at_k, k_alerts = precision_recall_at_k(y, p, frame["step"].values, CONFIG["alerts_per_day_capacity"])
    return {
        "model": label,
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision_at_k": p_at_k, "recall_at_k": r_at_k,
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts": tp + fp,
        "fraud_value_detected": amt[(pred == 1) & (y == 1)].sum(),
        "fraud_value_missed": amt[(pred == 0) & (y == 1)].sum(),
    }


# %% ---------------------------------------------------------------------------
# SECTION 8. MODEL TRAINING AND TUNING                         [Inst. Step 4]
# Report: PART VIII, Sections C and D
# -----------------------------------------------------------------------------
section("SECTION 8. MODEL TRAINING AND TUNING")


def build_model(name, params, y_train):
    if name == "LogisticRegression":
        return Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(class_weight="balanced", max_iter=3000, **params))])
    if name == "DecisionTree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RS, **params)
    if name == "RandomForest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RS, **params)
    if name == "XGBoost":
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        return XGBClassifier(tree_method="hist", eval_metric="aucpr", scale_pos_weight=spw,
                             n_jobs=-1, random_state=RS, **params)
    raise ValueError(name)


SEARCH_SPACE = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "DecisionTree": {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 20, 100, 500]},
    "RandomForest": {"n_estimators": [200, 400], "max_depth": [8, 12, 16, None],
                     "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.5]},
    "XGBoost": {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.8, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0], "min_child_weight": [1, 5, 10]},
}


def tune(name, cols, n_iter=None, tag=""):
    """Random search: fit on the training sample, score PR-AUC on validation.
    CHANGE (v2): n_iter and tag are new, so Section 10B can tune on Set D
    with more tries and save its own tuning log without overwriting this one."""
    X, y = fit_xy(cols)
    space = SEARCH_SPACE[name]
    n_total = int(np.prod([len(v) for v in space.values()]))
    n_iter = n_iter or CONFIG["n_search_iter"]
    candidates = list(ParameterSampler(space, n_iter=min(n_iter, n_total), random_state=RS))
    log, best, best_score = [], None, -1
    for params in candidates:
        m = build_model(name, params, y).fit(X, y)
        score = average_precision_score(val[TARGET], m.predict_proba(val[cols])[:, 1])
        log.append({**params, "val_pr_auc": score})
        if score > best_score:
            best, best_score, best_params = m, score, params
    save_table(pd.DataFrame(log).sort_values("val_pr_auc", ascending=False), f"08_tuning_{name}{tag}")
    print(f"  {name}: best validation PR-AUC {best_score:.4f} with {best_params}")
    return best, best_params, best_score


MODELS, BEST_PARAMS, VAL_SCORES, TUNE_TIME = {}, {}, {}, {}
for name in ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]:
    t0 = time.time()
    MODELS[name], BEST_PARAMS[name], VAL_SCORES[name] = tune(name, COLS_A)
    TUNE_TIME[name] = time.time() - t0
    print(f"  ({TUNE_TIME[name]:.0f}s)")
    a = ABBR[name]
    rep(f"TUNE_{a}_params", fmt_params(BEST_PARAMS[name]))
    rep(f"TUNE_{a}_val", f4(VAL_SCORES[name]))
    rep(f"TUNE_{a}_time", f"{TUNE_TIME[name]:.0f} seconds")
# RF can drag here. Scale down n_search_iter 


# %% ---------------------------------------------------------------------------
# SECTION 9. MODEL COMPARISON ON THE TEST SET                  [Inst. Step 4]
# Report: PART VIII, Sections E and G
# -----------------------------------------------------------------------------
section("SECTION 9. MODEL COMPARISON")

results = [evaluate(test[TARGET], test["isFlaggedFraud"], test, 0.5, "Rule benchmark (isFlaggedFraud)")]
TEST_PROBS = {}
for name, m in MODELS.items():
    TEST_PROBS[name] = m.predict_proba(test[COLS_A])[:, 1]
    results.append(evaluate(test[TARGET], TEST_PROBS[name], test, 0.5, name))
comparison = pd.DataFrame(results).set_index("model")
save_table(comparison.round(4), "09_model_comparison_test")
print(comparison[["pr_auc", "roc_auc", "precision", "recall", "f1", "false_positive_rate",
                  "precision_at_k", "recall_at_k", "alerts"]].round(4))
# CHANGE (v2): save every number in this table for the report.
for label, a in [("Rule benchmark (isFlaggedFraud)", "RULE")] + [(n, ABBR[n]) for n in MODELS]:
    r = comparison.loc[label]
    rep(f"CMP_{a}_prauc", f4(r["pr_auc"])); rep(f"CMP_{a}_roc", f4(r["roc_auc"]))
    rep(f"CMP_{a}_prec", f4(r["precision"])); rep(f"CMP_{a}_rec", f4(r["recall"]))
    rep(f"CMP_{a}_alerts", f_int(r["alerts"]))
rule_r = comparison.loc["Rule benchmark (isFlaggedFraud)"]
rep("RULE_TP", f_int(rule_r["tp"]))
rep("TEST_FRAUD_N", f_int(test[TARGET].sum()))
rep("RULE_VALUE_M", f"{rule_r['fraud_value_detected'] / 1e6:,.0f} million")
rep("TEST_FRAUD_VALUE_B", f"{test.loc[test[TARGET] == 1, 'amount'].sum() / 1e9:,.2f} billion")

# Curves for the deck
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name, p in TEST_PROBS.items():
    pr, rc, _ = precision_recall_curve(test[TARGET], p)
    fpr, tpr, _ = roc_curve(test[TARGET], p)
    axes[0].plot(rc, pr, label=f"{name} ({average_precision_score(test[TARGET], p):.3f})")
    axes[1].plot(fpr, tpr, label=name)
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].set_title("Precision-recall (test)")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate"); axes[1].set_title("ROC (test)")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
save_fig("09_pr_and_roc_curves")

# Model selection uses VALIDATION results, not test, to avoid choosing on the test set.
# Models within the tolerance of the best score are treated as tied. Among tied
# models, the first in the preference list wins (tree models first, so the
# SHAP explanations come from the same model that produces the risk score).
best_val = max(VAL_SCORES.values())
tied = [n for n, v in VAL_SCORES.items() if best_val - v <= CONFIG["selection_tolerance"]]
BEST_NAME = next(n for n in CONFIG["selection_preference"] if n in tied)
BEST = MODELS[BEST_NAME]
print(f"Models tied within {CONFIG['selection_tolerance']} of the best validation PR-AUC: {tied}")
print(f"Selected model: {BEST_NAME}")
rep("SEL_A_TIED_N", str(len(tied)))
rep("SEL_A_SPREAD", f"{max(VAL_SCORES.values()) - min(VAL_SCORES.values()):.4f}")
# NOTE TO SELF: Since performance was basically a wash, broke the tie using explainability and stress test resilience.

# Overfitting check: training vs validation PR-AUC
overfit = pd.DataFrame({
    "train_pr_auc": {n: average_precision_score(train_s[TARGET], m.predict_proba(train_s[COLS_A])[:, 1]) for n, m in MODELS.items()},
    "val_pr_auc": VAL_SCORES,
})
overfit["gap"] = overfit["train_pr_auc"] - overfit["val_pr_auc"]
save_table(overfit.round(4), "09_overfitting_check")
print(overfit.round(4))
rep("OVERFIT_MAX_GAP", f"{overfit['gap'].abs().max():.3f}")
# NOTE TO SELF: if the gap looks weird—undersampling skews the train score baseline, ++ focus magnitude.


# %% ---------------------------------------------------------------------------
# SECTION 10. STRESS TESTS E1 TO E6                            [Inst. Step 4/5]
# Report: PART VIII, Section F (answers RQ3)
# -----------------------------------------------------------------------------
section("SECTION 10. STRESS TESTS")

STRESS_SETUP = [
    ("E1", "A_full", "Full PaySim features (headline)"),
    ("E2", "A_no_recon", "Without balance reconciliation features"),
    ("E3", "B_interbank", "Without receiving balances (interbank view)"),
    ("E5", "A_strict", "Without reconciliation or sender balance fields"),
    ("E6", "T_transaction_only", "Transaction details only, no balance fields (realistic view)"),
]
# Notesss: Keeping XGBoost hyperparameters frozen across E1-E6 on purpose to isolate the feature set impact cleanly.
# changes between E1 and E6 is the feature set  (controlled test).
# CHANGE (v2): the catch is that E6 with Set A settings is under-tuned.
# Section 10B tunes all four models on Set D to fix that.
STRESS_MODELS = {}      # kept so Sections 11 to 13 can reuse them
stress = []
for code, set_name, desc in STRESS_SETUP:
    cols = FEATURE_SETS[set_name]
    if code == "E1":
        m = BEST                                   # same model as Section 9
    else:
        X, y = fit_xy(cols)
        m = build_model(BEST_NAME, BEST_PARAMS[BEST_NAME], y).fit(X, y)
    STRESS_MODELS[code] = (m, cols)
    r = evaluate(test[TARGET], m.predict_proba(test[cols])[:, 1], test, 0.5, f"{code} {set_name}")
    r["description"], r["n_features"] = desc, len(cols)
    stress.append(r)

# E4: same features as E1, random stratified split instead of time split
rand_train, rand_test = train_test_split(df, test_size=0.15, stratify=df[TARGET], random_state=RS)
for part in (rand_train, rand_test):
    part["high_value"] = (part["amount"] >= high_value_cut).astype("int8")
    part["amt_vs_type_median"] = np.log1p(part["amount"]) - np.log1p(part["type"].map(type_median).astype(float))
rand_s = make_training_sample(rand_train, CONFIG["legit_train_fraction"])
m4 = build_model(BEST_NAME, BEST_PARAMS[BEST_NAME], rand_s[TARGET]).fit(rand_s[COLS_A], rand_s[TARGET])
r4 = evaluate(rand_test[TARGET], m4.predict_proba(rand_test[COLS_A])[:, 1], rand_test, 0.5, "E4 A_full random split")
r4["description"], r4["n_features"] = "Full features, random split instead of time split", len(COLS_A)
stress.append(r4)
# NOTE TO SELF: E4 reuses the training-set thresholds from the time split for
# two features. That is a small shortcut and only affects this comparison.

stress_df = pd.DataFrame(stress).set_index("model").sort_index()
save_table(stress_df.round(4), "10_stress_tests")
print(stress_df[["n_features", "pr_auc", "recall", "precision", "false_positive_rate", "alerts"]].round(4))
sp = stress_df["pr_auc"]


def stress_row(code):
    return stress_df.loc[[i for i in stress_df.index if i.startswith(code + " ")][0]]


for code in ["E1", "E2", "E3", "E4", "E5", "E6"]:
    r = stress_row(code)
    rep(f"ST_{code}_prauc", f4(r["pr_auc"])); rep(f"ST_{code}_rec", f4(r["recall"]))
    rep(f"ST_{code}_prec", f4(r["precision"]))
# NOTE TO SELF: E1 vs E2 vs E5 vs E6 is the main critical-thinking result.
# Each time one source of the simulator pattern was removed, the model found
# another (sender balances in E2, receiving balances in E5). E6 removes them all. The
# larger the drop, the more the headline result depends on how the simulator
# creates fraud. Quote these numbers in PART XII ("Overstated Performance").


# %% ---------------------------------------------------------------------------
# SECTION 10B. MODEL COMPARISON ON FEATURE SET D (E6)          [Inst. Step 4]
# Report: PART VIII, Section G
# CHANGE (v2): this whole section is new.
# Why: on Set A every model scores about 1.000, so Section 9 cannot tell
# the models apart. Set D is the realistic view, and it is hard enough
# that the models should actually differ. So the real comparison is here.
# -----------------------------------------------------------------------------
section("SECTION 10B. MODEL COMPARISON ON FEATURE SET D")
COLS_D = FEATURE_SETS["T_transaction_only"]

MODELS_D, PARAMS_D, VAL_D, TIME_D = {}, {}, {}, {}
for name in ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]:
    t0 = time.time()
    MODELS_D[name], PARAMS_D[name], VAL_D[name] = tune(name, COLS_D, CONFIG["n_search_iter_realistic"], "_setD")
    TIME_D[name] = time.time() - t0
    a = ABBR[name]
    rep(f"E6T_{a}_params", fmt_params(PARAMS_D[name]))
    rep(f"E6T_{a}_val", f4(VAL_D[name]))
    rep(f"E6T_{a}_time", f"{TIME_D[name]:.0f} seconds")
# NOTE TO SELF: Random Forest is the slow one here. If it takes too long,
# drop n_search_iter_realistic to 12. Write the final number in the report.

# Test results, with a bootstrap interval and the capacity view
rows_d, PROBS_D = [], {}
for name, m in MODELS_D.items():
    p = m.predict_proba(test[COLS_D])[:, 1]
    PROBS_D[name] = p
    r = evaluate(test[TARGET], p, test, 0.5, name)
    lo, hi = bootstrap_pr_auc(test[TARGET], p, CONFIG["n_bootstrap"])
    pk, rk, _ = precision_recall_at_k(test[TARGET], p, test["step"].values, CONFIG["operating_capacity"])
    rows_d.append({"model": name, "val_pr_auc": VAL_D[name], "pr_auc": r["pr_auc"], "ci_low": lo, "ci_high": hi,
                   "roc_auc": r["roc_auc"], "precision_at_150": pk, "recall_at_150": rk,
                   "recall_05": r["recall"], "precision_05": r["precision"]})
    a = ABBR[name]
    rep(f"E6C_{a}_prauc", f4(r["pr_auc"])); rep(f"E6C_{a}_ci", f"{lo:.3f} to {hi:.3f}")
    rep(f"E6C_{a}_roc", f4(r["roc_auc"])); rep(f"E6C_{a}_pk", f_pct(pk)); rep(f"E6C_{a}_rk", f_pct(rk))
comparison_d = pd.DataFrame(rows_d).set_index("model")
save_table(comparison_d.round(4), "10b_model_comparison_setD")
print(comparison_d.round(4))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, p in PROBS_D.items():
    pr, rc, _ = precision_recall_curve(test[TARGET], p)
    ax.plot(rc, pr, label=f"{NICE[name]} ({average_precision_score(test[TARGET], p):.3f})")
ax.axhline(test[TARGET].mean(), ls="--", color="grey", lw=0.8, label="Fraud rate (no skill)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=8)
plt.title("Precision-recall on Feature Set D (test)")
save_fig("10b_E6_pr_curves")

# Selection: same rule as Section 9, and still on VALIDATION scores.
best_d = max(VAL_D.values())
tied_d = [n for n, v in VAL_D.items() if best_d - v <= CONFIG["selection_tolerance"]]
REAL_NAME = next(n for n in CONFIG["selection_preference"] if n in tied_d)
print(f"Set D models tied on validation: {tied_d}. Selected: {REAL_NAME}")
rep("E6_SELECTED", NICE[REAL_NAME])
rep("E6_TIED", ", ".join(NICE[n] for n in tied_d))
best_by_val = max(VAL_D, key=VAL_D.get)
if len(tied_d) == 1:
    rep("E6_SELECT_SENTENCE", f"{NICE[REAL_NAME]} had the highest validation PR-AUC, and no other model came "
                              f"within {CONFIG['selection_tolerance']} of it, so it was selected on that basis.")
else:
    rep("E6_SELECT_SENTENCE", f"{', '.join(NICE[n] for n in tied_d)} were within {CONFIG['selection_tolerance']} "
                              f"of the best validation PR-AUC, so they were treated as tied. {NICE[REAL_NAME]} was "
                              f"selected from the tied models because SHAP explains it directly.")

overfit_d = pd.DataFrame({
    "train_pr_auc": {n: average_precision_score(train_s[TARGET], m.predict_proba(train_s[COLS_D])[:, 1])
                     for n, m in MODELS_D.items()},
    "val_pr_auc": VAL_D,
})
overfit_d["gap"] = overfit_d["train_pr_auc"] - overfit_d["val_pr_auc"]
save_table(overfit_d.round(4), "10b_overfitting_check_setD")
rep("E6_MAX_GAP", f"{overfit_d['gap'].abs().max():.3f}")
rep("E6_MAX_GAP_MODEL", NICE[overfit_d["gap"].abs().idxmax()])
# NOTE TO SELF: a big positive gap here (say above 0.10) means the model
# memorised training. Decision Tree with no depth limit is the usual suspect.

best_row_d = comparison_d.loc[REAL_NAME]
rep("E6_BEST_PRAUC", f4(best_row_d["pr_auc"]))
rep("E6_BEST_PRAUC3", f3(best_row_d["pr_auc"]))
rep("E6_BEST_R05", f4(best_row_d["recall_05"]))
rep("E6_BEST_P05", f4(best_row_d["precision_05"]))
rep("E6_BEST_CI", f"{best_row_d['ci_low']:.3f} to {best_row_d['ci_high']:.3f}")
rule_prauc = comparison.loc["Rule benchmark (isFlaggedFraud)", "pr_auc"]
rep("E6_VS_RULE", f"{best_row_d['pr_auc'] / rule_prauc:.0f}")
rep("E6_SHARED_PRAUC3", f3(stress_row("E6")["pr_auc"]))

# CHANGE (v2): Sections 11 to 13 now use the E6 model tuned on Set D.
# The shared-settings E6 stays in the stress test table as the controlled check.
STRESS_MODELS["E6"] = (MODELS_D[REAL_NAME], COLS_D)


# %% ---------------------------------------------------------------------------
# SECTION 11. CALIBRATION, RISK SCORE AND THRESHOLDS           [Inst. Step 4/5]
# Report: PART IX
# Run for three views so the trade-offs are visible:
#   E1 = headline model, E2 = without reconciliation features,
#   E5 = without sender balances, E6 = transaction details only (realistic)
# -----------------------------------------------------------------------------
section("SECTION 11. CALIBRATION, RISK SCORE AND THRESHOLDS")
VIEWS = ["E1", "E2", "E5", "E6"]
mid_cost = CONFIG["review_costs"][len(CONFIG["review_costs"]) // 2]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def risk_levels(scores):
    return pd.cut(scores, bins=[-1, 29, 69, 100], labels=["Low", "Medium", "High"])


SCORING = {}            # per view: calibrated test probabilities, chosen threshold, calibrators
CAL_NAMES = {"platt": "Platt scaling", "isotonic": "isotonic regression"}


def oof_calibrated(raw, y, method):
    """CHANGE (v2): 5-fold calibration inside the validation set.
    Each fold is calibrated on the other four folds. So the Brier score
    that picks the method is never measured on rows the calibrator saw."""
    y = np.asarray(y)
    out = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    for tr, te in skf.split(raw.reshape(-1, 1), y):
        if method == "platt":
            c = LogisticRegression(max_iter=1000).fit(logit(raw[tr]), y[tr])
            out[te] = c.predict_proba(logit(raw[te]))[:, 1]
        else:
            c = IsotonicRegression(out_of_bounds="clip").fit(raw[tr], y[tr])
            out[te] = c.predict(raw[te])
    return out


test_fraud_value = test.loc[test[TARGET] == 1, "amount"].sum()

for view in VIEWS:
    print(f"\n--- View {view} ---")
    m, cols = STRESS_MODELS[view]
    val_raw = m.predict_proba(val[cols])[:, 1]
    test_raw = m.predict_proba(test[cols])[:, 1]

    # CHANGE (v2): pick the calibration method on VALIDATION, not test.
    # In v1 I took the lowest TEST Brier score. That lets the test set make a
    # choice, so the test result is no longer an honest check.
    # Now: 5-fold Brier inside validation picks the method. Test Brier is
    # still printed. Looking at test is fine. Choosing with it is not.
    val_oof = {k: oof_calibrated(val_raw, val[TARGET].values, k) for k in ["platt", "isotonic"]}
    val_brier = {k: brier_score_loss(val[TARGET], v) for k, v in val_oof.items()}
    method = min(val_brier, key=val_brier.get)
    platt = LogisticRegression(max_iter=1000).fit(logit(val_raw), val[TARGET])
    iso = IsotonicRegression(out_of_bounds="clip").fit(val_raw, val[TARGET])
    options = {"uncalibrated": test_raw,
               "platt": platt.predict_proba(logit(test_raw))[:, 1],
               "isotonic": iso.predict(test_raw)}
    cal_table = pd.DataFrame({k: {"brier_val_5fold": val_brier.get(k, np.nan),
                                  "brier_test": brier_score_loss(test[TARGET], v),
                                  "pr_auc_test": average_precision_score(test[TARGET], v)}
                              for k, v in options.items()}).T
    save_table(cal_table.round(5), f"11_{view}_calibration_comparison")
    p_cal = options[method]
    print(f"Calibration method (picked on validation): {method}")
    rep(f"CAL_{view}", CAL_NAMES[method])

    # Reliability chart
    plt.figure(figsize=(5.5, 5))
    bins = np.linspace(0, 1, 11)
    for k, v in options.items():
        idx = np.clip(np.digitize(v, bins) - 1, 0, 9)
        used = [b for b in range(10) if (idx == b).sum() > 0]
        plt.plot([v[idx == b].mean() for b in used], [test[TARGET].values[idx == b].mean() for b in used],
                 marker="o", label=k)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("Predicted probability"); plt.ylabel("Observed fraud rate")
    plt.title(f"Calibration chart, {view} (test)"); plt.legend()
    save_fig(f"11_{view}_calibration_chart")

    # Risk score and score-based levels
    score = np.round(p_cal * 100).astype(int)
    lv = pd.DataFrame({"risk_level": risk_levels(score), TARGET: test[TARGET].values})
    levels = lv.groupby("risk_level", observed=False)[TARGET].agg(transactions="size", fraud="sum")
    levels["fraud_rate_pct"] = (levels["fraud"] / levels["transactions"].replace(0, np.nan) * 100).round(3)
    save_table(levels, f"11_{view}_risk_levels")
    print(levels)

    # Threshold analysis on TEST. This table is for reporting only.
    rows = []
    for t in CONFIG["thresholds"]:
        r = evaluate(test[TARGET], p_cal, test, t, f"t={t}")
        row = {"threshold": t, "recall": r["recall"], "precision": r["precision"], "alerts": r["alerts"],
               "false_alerts": r["fp"], "fraud_value_detected": r["fraud_value_detected"],
               "fraud_value_missed": r["fraud_value_missed"],
               "fraud_value_share": r["fraud_value_detected"] / max(test_fraud_value, 1)}
        for c in CONFIG["review_costs"]:
            row[f"expected_cost_review_{c}"] = r["fraud_value_missed"] + r["alerts"] * c
        rows.append(row)
    thr = pd.DataFrame(rows).set_index("threshold")
    save_table(thr.round(4), f"11_{view}_threshold_analysis")
    print(thr[["recall", "precision", "alerts", "false_alerts"]].round(4))
    test_cost_t = float(thr[f"expected_cost_review_{mid_cost}"].idxmin())

    # CHANGE (v2): the cost-based threshold is now found on VALIDATION, using
    # the out-of-fold calibrated scores from above. v1 found it on test.
    val_cost = {}
    for t in CONFIG["thresholds"]:
        r = evaluate(val[TARGET], val_oof[method], val, t, f"val t={t}")
        val_cost[t] = r["fraud_value_missed"] + r["alerts"] * mid_cost
    chosen_t = float(min(val_cost, key=val_cost.get))
    print(f"Lowest expected cost on validation (review cost {mid_cost}): threshold {chosen_t}")
    print(f"For comparison only, lowest expected cost on test: threshold {test_cost_t}")

    # Precision-recall trade-off chart for the deck
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(thr.index, thr["recall"], marker="o", label="Recall", color="#0C3512")
    ax.plot(thr.index, thr["precision"], marker="o", label="Precision", color="#9B1D1D")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Share"); ax.set_ylim(0, 1.05)
    ax2 = ax.twinx(); ax2.bar(thr.index, thr["alerts"], width=0.03, alpha=0.25, color="grey", label="Alerts")
    ax2.set_ylabel("Alerts")
    ax.legend(loc="lower left"); plt.title(f"Threshold trade-off, {view}")
    save_fig(f"11_{view}_threshold_tradeoff")

    # Capacity-based view at several staffing levels
    cap_rows = []
    for k in CONFIG["capacity_options"]:
        pk, rk, nk = precision_recall_at_k(test[TARGET], p_cal, test["step"], k)
        a_k = capacity_alerts(p_cal, test["step"].values, k)
        v_k = test["amount"].values[(a_k == 1) & (test[TARGET].values == 1)].sum() / max(test_fraud_value, 1)
        cap_rows.append({"alerts_per_day": k, "precision_at_k": pk, "recall_at_k": rk, "total_alerts": nk,
                         "fraud_value_share": v_k})
    cap = pd.DataFrame(cap_rows).set_index("alerts_per_day")
    save_table(cap.round(4), f"11_{view}_capacity_view")
    print(cap.round(4))

    SCORING[view] = {"p_cal": p_cal, "threshold": chosen_t, "test_cost_threshold": test_cost_t, "method": method,
                     "platt": platt, "iso": iso, "capacity": cap, "thresholds": thr, "levels": levels}

days_in_test = test["step"].nunique() / 24
fraud_per_day = test[TARGET].sum() / days_in_test
print(f"\nTest period: about {days_in_test:.1f} days, {fraud_per_day:.0f} fraud cases per day.")
# Reality check for bizz team: if daily fraud outstrips our review capacity limit, 100% recall is mathematically impossible regardless of model quality.
# The review costs and capacities are assumptions. Label as illustrative.

# CHANGE (v2): numbers for PART IX of the report. These are all for E6,
# the realistic view, because that is the one the report discusses.
e6s = SCORING["E6"]
rep("FRAUD_PER_DAY", f_int(fraud_per_day))
rep("TEST_DAYS", f"{days_in_test:.1f}")
rep("VAL_COST_T", f"{SCORING['E6']['threshold']:.2f}")
rep("TEST_COST_T", f"{SCORING['E6']['test_cost_threshold']:.2f}")
rep("E6_MEAN_SCORE", f"{e6s['p_cal'].mean() * 100:.2f}%")
rep("TEST_FRAUD_RATE", f"{test[TARGET].mean() * 100:.2f}%")
rep("VAL_FRAUD_RATE", f"{val[TARGET].mean() * 100:.2f}%")
rep("TRAIN_FRAUD_RATE", f"{train[TARGET].mean() * 100:.2f}%")
for lvl in ["Low", "Medium", "High"]:
    rep(f"RL_{lvl}_n", f_int(e6s["levels"].loc[lvl, "transactions"]))
    rep(f"RL_{lvl}_fraud", f_int(e6s["levels"].loc[lvl, "fraud"]))
    fr = e6s["levels"].loc[lvl, "fraud_rate_pct"]
    rep(f"RL_{lvl}_rate", "n/a" if pd.isna(fr) else f"{fr:.2f}%")
for k in CONFIG["capacity_options"]:
    c = e6s["capacity"].loc[k]
    rep(f"CAP_{k}_rec", f_pct(c["recall_at_k"])); rep(f"CAP_{k}_prec", f_pct(c["precision_at_k"]))
    rep(f"CAP_{k}_alerts", f_int(c["total_alerts"])); rep(f"CAP_{k}_val", f_pct(c["fraud_value_share"]))
for t in [0.01, 0.02, 0.05, 0.10, 0.50, 0.90]:
    row = e6s["thresholds"].loc[t]
    tag = f"{int(round(t * 100)):03d}"
    rep(f"TH_{tag}_rec", f_pct(row["recall"])); rep(f"TH_{tag}_prec", f_pct(row["precision"]))
    rep(f"TH_{tag}_day", f_int(row["alerts"] / days_in_test)); rep(f"TH_{tag}_fp", f_int(row["false_alerts"]))
    rep(f"TH_{tag}_val", f_pct(row["fraud_value_share"]))


# %% ---------------------------------------------------------------------------
# SECTION 12. EXPLAINABILITY (SHAP)                            [Inst. Step 5]
# Report: PART X
# SHAP runs on the selected model (E1) and on the E2 model, so I can compare
# what the model relies on with and without the reconciliation features.
# -----------------------------------------------------------------------------
section("SECTION 12. EXPLAINABILITY")
import shap

REASON_TEXT = {
    "orig_emptied": "Sender account emptied", "errorBalanceOrig": "Sender balance does not reconcile",
    "errorBalanceDest": "Receiving balance does not reconcile", "dest_zero_after": "Receiving balance stays at zero",
    "log_amount": "Large amount", "amt_to_orig_bal": "Amount large relative to sender balance",
    "amt_to_dest_bal": "Amount large relative to receiving balance", "high_value": "High-value transaction",
    "amt_vs_type_median": "Amount well above usual for this type", "oldbalanceOrg": "Sender balance level",
    "newbalanceOrig": "Sender balance after transfer", "oldbalanceDest": "Receiving balance level",
    "newbalanceDest": "Receiving balance after transfer", "hour": "Time of day",
    "period_night": "Night-time transaction", "period_morning": "Morning transaction",
    "period_afternoon": "Afternoon transaction", "type_TRANSFER": "Transaction type",
    "dest_prior_count": "Receiving account history", "log_dest_prior_amount": "Amount previously received by account",
    "dest_is_new": "New receiving account",
}

shap_base = test.sample(min(CONFIG["shap_sample"], len(test)), random_state=RS)
shap_base = pd.concat([shap_base, test[test[TARGET] == 1].sample(min(1000, int(test[TARGET].sum())), random_state=RS)])
shap_base = shap_base[~shap_base.index.duplicated()]


def shap_values_for(model, X):
    """Tree models use TreeExplainer. Logistic Regression uses LinearExplainer.
    CHANGE (v2): XGBoost now uses its own built-in SHAP values (pred_contribs).
    On my Mac, shap 0.49.1 (last version for Python 3.10) could not read
    models from xgboost 3.x. It crashed with "could not convert string to
    float: '[5E-1]'". XGBoost computes the same TreeSHAP values itself, so
    the numbers do not change. Only the tool that computes them changed."""
    if isinstance(model, XGBClassifier):
        import xgboost as xgb
        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contribs[:, :-1]     # last column is the bias term, not a feature
    if isinstance(model, Pipeline):
        sc, lr = model.named_steps["sc"], model.named_steps["lr"]
        Xs = sc.transform(X)
        bg = sc.transform(train_s[X.columns].sample(min(1000, len(train_s)), random_state=RS))
        return shap.LinearExplainer(lr, bg).shap_values(Xs)
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


SHAP_IMPORTANCE, ALERT_LISTS = {}, {}
# NOTE TO SELF: since v2 the E6 model might not be XGBoost (Section 10B picks
# it). shap_values_for() already handles Logistic Regression and tree models.
# Random Forest with deep trees is slow in TreeExplainer. If it crawls,
# lower shap_sample in CONFIG to 2000.
for view in ["E1", "E2", "E5", "E6"]:
    m, cols = STRESS_MODELS[view]
    X = shap_base[cols]
    sv = shap_values_for(m, X)
    shap.summary_plot(sv, X, show=False, max_display=15)
    save_fig(f"12_{view}_shap_summary")
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
    save_fig(f"12_{view}_shap_importance_bar")
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols, name="mean_abs_shap").sort_values(ascending=False)
    SHAP_IMPORTANCE[view] = imp
    save_table(imp.to_frame(), f"12_{view}_shap_global_importance")
    for feat in imp.index[:3]:
        shap.dependence_plot(feat, sv, X, show=False, interaction_index=None)
        save_fig(f"12_{view}_shap_dependence_{feat}")

    # Plain-language reasons (top 3 positive contributions) for a sample alert list.
    # The risk score and the reasons come from the SAME model.
    def reasons(row_shap, top=3):
        s_ = pd.Series(row_shap, index=cols)
        s_ = s_[s_ > 0].sort_values(ascending=False).head(top)
        return "; ".join(REASON_TEXT.get(f, f) for f in s_.index)

    pos = test.index.get_indexer(shap_base.index)
    alerts = shap_base[["step", "type", "amount", TARGET]].copy()
    alerts["risk_score"] = np.round(SCORING[view]["p_cal"][pos] * 100).astype(int) if view in SCORING else np.nan
    alerts["priority"] = risk_levels(alerts["risk_score"])
    alerts["reasons"] = [reasons(r) for r in sv]
    alerts = alerts.sort_values("risk_score", ascending=False)
    save_table(alerts.head(30), f"12_{view}_sample_alert_list")
    ALERT_LISTS[view] = alerts.head(30)
    print(f"\nTop alerts, {view}:")
    print(alerts.head(8)[["type", "amount", "risk_score", TARGET, "reasons"]])

shap_compare = pd.DataFrame(SHAP_IMPORTANCE).fillna(0).sort_values("E1", ascending=False)
save_table(shap_compare.round(5), "12_shap_comparison_all_views")

# CHANGE (v2): top three features per view go straight into the report table.
# In v1 I typed these in, which is how report and code drift apart.
def label(feat):
    return REASON_TEXT.get(feat, feat)


for view in ["E1", "E2", "E5", "E6"]:
    top = SHAP_IMPORTANCE[view].index[:3]
    rep(f"SHAP_{view}_TOP3", "; ".join([label(top[0])] + [label(f)[0].lower() + label(f)[1:] for f in top[1:]]))
imp6 = SHAP_IMPORTANCE["E6"]
time_feats = ["hour", "period_night", "period_morning", "period_afternoon"]
time_rank = min(int(np.where(imp6.index == f)[0][0]) + 1 for f in time_feats if f in imp6.index)
rep("SHAP_E6_TIME_RANK", str(time_rank))
rep("SHAP_E6_NFEAT", str(len(imp6)))
# Most common reasons in the E6 sample alert list (top 30 by score)
reason_counts = pd.Series([x for s in ALERT_LISTS["E6"]["reasons"] for x in str(s).split("; ") if x]).value_counts()
top_r = [f'"{r[0].lower() + r[1:]}"' for r in reason_counts.index[:3]]
rep("SHAP_E6_REASONS", ", ".join(top_r[:-1]) + " and " + top_r[-1] if len(top_r) > 1 else (top_r[0] if top_r else "none"))
# Logistic Regression coefficients (standardized), for the report
lr_coef = pd.Series(MODELS["LogisticRegression"].named_steps["lr"].coef_[0], index=COLS_A,
                    name="coefficient").sort_values(key=np.abs, ascending=False)
save_table(lr_coef.to_frame(), "12_logistic_coefficients")
# NOTE TO SELF: compare importance across E1, E2, E5 and E6. If E6 still leans
# heavily on hour of day, that is the simulator spreading fraud evenly across
# hours while legitimate activity peaks in the daytime. Discuss it in PART XI.
# Compare E1 and E2 importance. E1 should lean on the simulator
# patterns (emptied account, reconciliation). E2 shows what the model uses
# when those are gone. That contrast is a strong slide for the technical deck.


# %% ---------------------------------------------------------------------------
# SECTION 13. BIAS AND FAIRNESS REVIEW                         [Inst. Step 5]
# Report: PART XI
# Run for E1, E2, E5 and E6. E1 has almost no false alerts, so E5 and E6 are
# where the group comparison actually tells me something.
# -----------------------------------------------------------------------------
section("SECTION 13. BIAS AND FAIRNESS REVIEW")

# PaySim has no demographic fields. The review uses transaction groups instead.
#   - Alert rate per group (demographic parity view)
#   - False positive rate and true positive rate per group (equalized odds view)
#   - Ratio of each group's FPR to the overall FPR (disparity ratio)
groups = pd.DataFrame(index=test.index)
groups["type"] = test["type"].astype(str)
groups["amount_band"] = pd.qcut(test["amount"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
groups["sender_balance_band"] = pd.cut(test["oldbalanceOrg"], [-1, 0, 10_000, 100_000, np.inf],
                                       labels=["Zero", "Up to 10k", "10k to 100k", "Over 100k"])
groups["period"] = np.select([test.hour.between(0, 5), test.hour.between(6, 11), test.hour.between(12, 17)],
                             ["Night", "Morning", "Afternoon"], "Evening")
GROUP_COLS = ["type", "amount_band", "sender_balance_band", "period"]


def group_metrics(y_all, alert_all, group_col):
    out = []
    frame = pd.DataFrame({"y": y_all, "a": alert_all, "g": groups[group_col].values})
    overall_fpr = ((frame.a == 1) & (frame.y == 0)).sum() / max((frame.y == 0).sum(), 1)
    for g, d in frame.groupby("g", observed=True):
        tn, fp, fn, tp = confusion_matrix(d.y, d.a, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        out.append({"group_by": group_col, "group": str(g), "transactions": len(d), "fraud": int(d.y.sum()),
                    "fraud_rate": d.y.mean(), "alert_rate": d.a.mean(), "fpr": fpr,
                    "tpr": tp / max(tp + fn, 1), "fnr": fn / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                    "fpr_ratio_vs_overall": fpr / overall_fpr if overall_fpr > 0 else np.nan})
    return pd.DataFrame(out)


FAIRNESS, BIAS_ROWS = {}, {}


def alert_rule(view, rule):
    """CHANGE (v2): one function for both alert rules.
    rule == "capacity" -> top k per day. This is the main rule in the report.
    rule == a number   -> fixed threshold on the calibrated score (comparison).
    In v1 the bias review used the cost-based threshold from Section 11,
    while the report showed 0.05. Now both use this function."""
    p = SCORING[view]["p_cal"]
    if rule == "capacity":
        return capacity_alerts(p, test["step"].values, CONFIG["operating_capacity"])
    return (p >= rule).astype(int)


def fairness_summary(alert):
    bias = pd.concat([group_metrics(test[TARGET].values, alert, c) for c in GROUP_COLS])
    summary = bias.groupby("group_by").agg(
        demographic_parity_difference=("alert_rate", lambda s: s.max() - s.min()),
        fraud_rate_difference=("fraud_rate", lambda s: s.max() - s.min()),
        equalized_odds_fpr_gap=("fpr", lambda s: s.max() - s.min()),
        equalized_odds_tpr_gap=("tpr", lambda s: s.max() - s.min()),
        max_fpr_ratio=("fpr_ratio_vs_overall", "max"),
        min_fpr_ratio=("fpr_ratio_vs_overall", "min"),
    )
    return bias, summary


RULES = {"E1": ["capacity"], "E2": ["capacity"], "E5": ["capacity"],
         "E6": ["capacity"] + CONFIG["compare_thresholds"]}
for view, rules in RULES.items():
    for rule in rules:
        tag = "cap" if rule == "capacity" else f"t{rule}"
        bias, summary = fairness_summary(alert_rule(view, rule))
        save_table(bias.set_index(["group_by", "group"]).round(4), f"13_{view}_{tag}_bias_review")
        save_table(summary.round(4), f"13_{view}_{tag}_fairness_summary")
        FAIRNESS[(view, tag)], BIAS_ROWS[(view, tag)] = summary, bias
        print(f"\nFairness summary, {view}, rule {tag}:")
        print(summary.round(4))

# CHANGE (v2): numbers and plain-language verdicts for PART XI.
GROUP_KEY = {"type": "TYPE", "amount_band": "AMT", "sender_balance_band": "BAL", "period": "TIME"}


def gap_verdict(alert_gap, fraud_gap):
    """Compare the alert-rate gap with the fraud-rate gap for one grouping."""
    if alert_gap <= fraud_gap + 0.005:
        return "Within the fraud-rate gap"
    return f"{(alert_gap - fraud_gap) * 100:.1f} points above the fraud-rate gap"


for tag, prefix in [("cap", "BIAS"), ("t0.05", "BT005"), ("t0.01", "BT001")]:
    s = FAIRNESS[("E6", tag)]
    for g, k in GROUP_KEY.items():
        rep(f"{prefix}_{k}_alert", f3(s.loc[g, "demographic_parity_difference"]))
        rep(f"{prefix}_{k}_fraud", f3(s.loc[g, "fraud_rate_difference"]))
        rep(f"{prefix}_{k}_fpr", f3(s.loc[g, "equalized_odds_fpr_gap"]))
        rep(f"{prefix}_{k}_tpr", f3(s.loc[g, "equalized_odds_tpr_gap"]))
        rep(f"{prefix}_{k}_verdict", gap_verdict(s.loc[g, "demographic_parity_difference"],
                                                 s.loc[g, "fraud_rate_difference"]))

b6 = BIAS_ROWS[("E6", "cap")]
b6 = b6.reset_index(drop=True)
i_top = b6["fpr_ratio_vs_overall"].idxmax()
GROUP_NAME = {"type": "transaction type", "amount_band": "amount range",
              "sender_balance_band": "sender balance range", "period": "time of day"}
rep("BIAS_TOP_FPR_GROUP", f"{b6.loc[i_top, 'group']} ({GROUP_NAME[b6.loc[i_top, 'group_by']]})")
rep("BIAS_TOP_FPR_RATIO", f"{b6.loc[i_top, 'fpr_ratio_vs_overall']:.1f}")
per = b6[b6["group_by"] == "period"].set_index("group")
rep("BIAS_TOP_FPR_PERIOD", per["fpr"].idxmax())
rep("BIAS_NIGHT_FPR_RATIO", f"{per.loc['Night', 'fpr_ratio_vs_overall']:.1f}" if "Night" in per.index else "n/a")
enough = b6[b6["fraud"] >= 30]
i_low = enough["tpr"].idxmin()
rep("BIAS_LOW_TPR_GROUP", f"{enough.loc[i_low, 'group']} ({GROUP_NAME[enough.loc[i_low, 'group_by']]})")
rep("BIAS_LOW_TPR", f_pct(enough.loc[i_low, "tpr"]))
i_high = enough["tpr"].idxmax()
rep("BIAS_HIGH_TPR_GROUP", f"{enough.loc[i_high, 'group']} ({GROUP_NAME[enough.loc[i_high, 'group_by']]})")
rep("BIAS_HIGH_TPR", f_pct(enough.loc[i_high, "tpr"]))
# NOTE: I only compare groups with at least 30 fraud cases for the
# lowest and highest detection rate. A group with 3 fraud cases can show 0%
# or 100% by luck, and that would be a silly thing to put in a report.

# ALSO: read demographic parity next to fraud_rate_difference. If a group
# gets more alerts because it simply has more fraud, that is not unfair treatment.
# The equalized odds gaps (FPR and TPR) are the better test.
# A ratio outside 0.8 to 1.25 is a common flag. This'll be treated as a discussion point, not a pass/fail test.
# Mitigation options to discuss in PART XI (Step 5 asks for feasible ones):
#   1. Review the threshold for any group with a high false positive rate.
#   2. Reweight training rows for over-flagged groups and compare results.
#   3. Drop or cap features that drive the gap if they add little accuracy.
#   4. Monitor group error rates monthly once real data is used.
#   5. Keep a human reviewer on every alert; the model never blocks transactions.
# Merchant vs non-merchant could not be reviewed, because the scope filter
# leaves no merchant receivers. State this in the report.

# 13.2 Mitigation test M1: drop the time-of-day features
# CHANGE (v2): new. In v1 the mitigations were only ideas. Step 5 asks for
# "feasible" mitigations, so I test the simplest one for real.
# Idea: the time features drive the night-time false alerts. Retrain the
# selected E6 model without them, keep everything else the same, and
# compare. The capacity rule only needs the ranking, so no calibration here.
TIME_FEATS = ["hour", "period_night", "period_morning", "period_afternoon"]
cols_m1 = [c for c in COLS_D if c not in TIME_FEATS]
X_m1, y_m1 = fit_xy(cols_m1)
m1 = build_model(REAL_NAME, PARAMS_D[REAL_NAME], y_m1).fit(X_m1, y_m1)
p_m1 = m1.predict_proba(test[cols_m1])[:, 1]
p_base = SCORING["E6"]["p_cal"]
m1_rows = []
for label_, p in [("base", p_base), ("new", p_m1)]:
    alert = capacity_alerts(p, test["step"].values, CONFIG["operating_capacity"])
    _, summ = fairness_summary(alert)
    pk, rk, _ = precision_recall_at_k(test[TARGET], p, test["step"].values, CONFIG["operating_capacity"])
    m1_rows.append({"version": label_, "pr_auc": average_precision_score(test[TARGET], p),
                    "recall_at_150": rk, "precision_at_150": pk,
                    "time_alert_gap": summ.loc["period", "demographic_parity_difference"],
                    "time_fpr_gap": summ.loc["period", "equalized_odds_fpr_gap"],
                    "time_tpr_gap": summ.loc["period", "equalized_odds_tpr_gap"]})
m1_table = pd.DataFrame(m1_rows).set_index("version")
save_table(m1_table.round(4), "13_mitigation_M1_no_time_features")
print(m1_table.round(4))
for v in ["base", "new"]:
    r = m1_table.loc[v]
    V = v.upper()
    rep(f"M1_{V}_PRAUC", f4(r["pr_auc"])); rep(f"M1_{V}_REC", f_pct(r["recall_at_150"]))
    rep(f"M1_{V}_PREC", f_pct(r["precision_at_150"])); rep(f"M1_{V}_ALERTGAP", f3(r["time_alert_gap"]))
    rep(f"M1_{V}_FPRGAP", f3(r["time_fpr_gap"])); rep(f"M1_{V}_TPRGAP", f3(r["time_tpr_gap"]))
d_prauc = m1_table.loc["base", "pr_auc"] - m1_table.loc["new", "pr_auc"]
fpr_better = m1_table.loc["new", "time_fpr_gap"] < m1_table.loc["base", "time_fpr_gap"]
if fpr_better and d_prauc <= 0.02:
    verdict = (f"Removing the time features narrowed the false alert gap by time of day, and PR-AUC changed by only "
               f"{d_prauc:+.3f}. This is a workable mitigation and is recommended for the next iteration.")
elif fpr_better:
    verdict = (f"Removing the time features narrowed the false alert gap by time of day, but PR-AUC fell by "
               f"{d_prauc:.3f}. The trade-off between fairness and fraud caught is a business decision.")
else:
    verdict = ("Removing the time features did not narrow the false alert gap by time of day. The other features "
               "carry the same pattern, so threshold review and monthly monitoring remain the main mitigations.")
rep("M1_VERDICT", verdict)
# TO SELF: whatever the verdict says, report it as is. A mitigation that did not work is still a valid finding.... :)


# %% --------------------------------------------------------------------------
# SECTION 14. LIMITATIONS CHECKLIST                            [Inst. Step 5]
# Report: PART XII
# -----------------------------------------------------------------------------
section("SECTION 14. LIMITATIONS CHECKLIST")
sp = stress_df["pr_auc"]
pick = lambda code: float(sp[[i for i in sp.index if i.startswith(code)][0]])
limits = pd.DataFrame([
    ["Class imbalance", f"Fraud share: train {train[TARGET].mean() * 100:.3f}%, test {test[TARGET].mean() * 100:.3f}%"],
    ["Overstated performance", f"PR-AUC E1 {pick('E1'):.3f}, E2 {pick('E2'):.3f}, E5 {pick('E5'):.3f}, "
                               f"E6 shared settings {pick('E6'):.3f}, E6 tuned ({REAL_NAME}) {best_row_d['pr_auc']:.3f}"],
    ["Interbank data gap", f"E3 PR-AUC {pick('E3'):.3f} vs E1 {pick('E1'):.3f}"],
    ["Split method effect", f"E4 PR-AUC {pick('E4'):.3f} vs E1 {pick('E1'):.3f}"],
    ["Overfitting", f"Largest train/val gap {overfit['gap'].abs().max():.3f} ({overfit['gap'].abs().idxmax()})"],
    ["Review capacity", f"About {fraud_per_day:.0f} fraud cases per test day vs capacity options {CONFIG['capacity_options']}"],
    ["Leakage", "isFlaggedFraud used as benchmark only; post-transaction balances documented"],
    ["Time coverage", f"{checks['max_step']} hourly steps (about one month)"],
    ["Calibration drift", f"Fraud rate: validation {val[TARGET].mean() * 100:.2f}%, test {test[TARGET].mean() * 100:.2f}%"],
    ["Fairness mitigation", REPORT["M1_VERDICT"]],
], columns=["limitation", "evidence_from_this_run"]).set_index("limitation")
save_table(limits, "14_limitations_evidence")
print(limits.to_string())


# %% ---------------------------------------------------------------------------
# SECTION 15. OUTPUTS FOR THE TWO DECKS                         [Inst.Step 6]
# Report: PART XIII and PART XIV
# -----------------------------------------------------------------------------
section("SECTION 15. DECK OUTPUTS")
best_row = comparison.loc[BEST_NAME]
rule_row = comparison.loc["Rule benchmark (isFlaggedFraud)"]


def view_summary(view):
    t = SCORING[view]["threshold"]
    r = evaluate(test[TARGET], SCORING[view]["p_cal"], test, t, view)
    return {"threshold": t, "pr_auc": round(r["pr_auc"], 4), "recall": round(r["recall"], 4),
            "precision": round(r["precision"], 4), "alerts": int(r["alerts"]), "false_alerts": int(r["fp"]),
            "fraud_value_detected": float(r["fraud_value_detected"]),
            "capacity": SCORING[view]["capacity"].round(4).to_dict(orient="index")}


deck = {
    "technical_deck": {
        "selected_model": BEST_NAME,
        "realistic_model_setD": REAL_NAME,
        "setD_comparison": comparison_d.round(4).to_dict(orient="index"),
        "mitigation_M1": m1_table.round(4).to_dict(orient="index"),
        "tied_models_on_validation": tied,
        "test_pr_auc_by_model": comparison["pr_auc"].round(4).to_dict(),
        "stress_tests_pr_auc": stress_df["pr_auc"].round(4).to_dict(),
        "top_features_E1": SHAP_IMPORTANCE["E1"].head(5).round(4).to_dict(),
        "top_features_E2": SHAP_IMPORTANCE["E2"].head(5).round(4).to_dict(),
        "top_features_E5": SHAP_IMPORTANCE["E5"].head(5).round(4).to_dict(),
        "top_features_E6": SHAP_IMPORTANCE["E6"].head(5).round(4).to_dict(),
        "figures": sorted(os.listdir(CONFIG["figures_dir"])),
    },
    "business_deck": {
        "fraud_value_in_test": float(test.loc[test[TARGET] == 1, "amount"].sum()),
        "rule_benchmark": {"alerts": int(rule_row["alerts"]), "fraud_caught": int(rule_row["tp"]),
                           "fraud_value_detected": float(rule_row["fraud_value_detected"])},
        "headline_view_E1": view_summary("E1"),
        "no_recon_view_E2": view_summary("E2"),
        "no_sender_balance_view_E5": view_summary("E5"),
        "realistic_view_E6": view_summary("E6"),
        "fraud_cases_per_day_in_test": round(float(fraud_per_day), 1),
        "operating_rule": f"Top {CONFIG['operating_capacity']} scores per day (capacity-based)",
        "note": "Simulated data, assumed costs and assumed capacity. Not an estimate of Agribank results.",
    },
}
with open(Path(CONFIG["tables_dir"]) / "15_deck_summary.json", "w") as f:
    json.dump(deck, f, indent=2, default=str)
print(json.dumps(deck, indent=2, default=str))
# NOTE: on my business deck, show E1 as "what the simulator allows"
# and E6 as "a more realistic expectation". Not to lead with 99.9%.
# Technical deck: problem, data, EDA, features, models,
# comparison, stress tests, SHAP E1 vs E2, bias, limitations.
# Business deck: the problem, what the score does, fraud value
# caught vs the rule, alert workload and staffing, risks, next steps.
# Amounts are simulated units. Do not present these as pesos.


# %% ---------------------------------------------------------------------------
# SECTION 16. SAVE MODELS AND CONFIGURATION                 [Inst. Step 7]
# -----------------------------------------------------------------------------
section("SECTION 16. SAVE MODELS AND CONFIGURATION")
for name, m in MODELS.items():
    joblib.dump(m, Path(CONFIG["models_dir"]) / f"{name}.joblib")
for view in ["E2", "E5"]:
    joblib.dump(STRESS_MODELS[view][0], Path(CONFIG["models_dir"]) / f"{BEST_NAME}_{view}.joblib")
# CHANGE (v2): save the four Set D models too. The E6 model in the report
# is the one named in run_config.json.
for name, m in MODELS_D.items():
    joblib.dump(m, Path(CONFIG["models_dir"]) / f"{name}_setD.joblib")
joblib.dump({v: {"platt": SCORING[v]["platt"], "isotonic": SCORING[v]["iso"], "method": SCORING[v]["method"],
                 "threshold": SCORING[v]["threshold"]} for v in SCORING},
            Path(CONFIG["models_dir"]) / "calibrators.joblib")
run_config = {
    "config": CONFIG, "feature_sets": FEATURE_SETS, "dropped_features": weak,
    "best_params": BEST_PARAMS, "selected_model": BEST_NAME, "tied_models": tied,
    "best_params_setD": PARAMS_D, "realistic_model_setD": REAL_NAME, "tied_models_setD": tied_d,
    "reference_stats": REFERENCE_STATS,
    "time_split_steps": {"train_before": int(cut_train), "val_before": int(cut_val)},
    "chosen_thresholds": {v: SCORING[v]["threshold"] for v in SCORING},
    "library_versions": {"pandas": pd.__version__, "numpy": np.__version__,
                         "sklearn": __import__("sklearn").__version__,
                         "xgboost": __import__("xgboost").__version__, "shap": shap.__version__},
}
with open(Path(CONFIG["models_dir"]) / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2, default=str)
print("Saved models, calibrators and run_config.json")

# CHANGE (v2): write requirements.txt from the versions this run actually used.
# In v1 I planned to type them by hand. This way the repo always matches.
from importlib.metadata import version as pkg_version, PackageNotFoundError


def pinned(pkg):
    try:
        return f"{pkg}=={pkg_version(pkg)}"
    except PackageNotFoundError:
        return pkg


req = [pinned(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "shap", "matplotlib", "joblib",
                           "python-docx"]]
if CONFIG["use_smote"]:
    req.append(pinned("imbalanced-learn"))
Path("requirements.txt").write_text("\n".join(req) + "\n")
print("Wrote requirements.txt:", req)
rep("LIB_VERSIONS", "; ".join(req[:6]))
rep("SEED", str(RS))
# REMEMBER self before pushing to GitHub:
#   - Add data/raw/ to .gitignore. Put the Kaggle link in the README instead kasi this is big , piblic file naman. 
#   - Saved models can be large. Add models/*.joblib to .gitignore if GitHub, ok
#     rejects files over 100 MB, and say in the README how to regenerate them.
#   - README: problem, dataset source, how to run, folder layout, key results,
#     limitations. Link the final report PDF.
#   - requirements.txt: pin the versions printed in run_config.json.
#   - Commit in small steps with clear messages (e.g. "Add EDA figures"),
#     not one large upload at the end para sa ghub f limit. 


# %% ----------------------------------------------------------------------------
# SECTION 17. CHECK THE REPORT'S CLAIMS AND FILL IN THE NUMBERS [Instr. Step 6/7]
# CHANGE (v2): new section.
# Two problems in v1: (1) numbers in the report were typed by hand, and
# (2) some sentences in the report are claims ("E2 barely changed") that
# could stop being true after a rerun. This section handles both.
# -------------------------------------------------------------------------------
section("SECTION 17. REPORT CLAIMS AND NUMBERS")

# 17.1 Claim checks. Each line is a sentence the report makes in words.
# If one says FAIL, should rewrite that sentence in the report.
sa = SHAP_IMPORTANCE["E1"]
CLAIMS = [
    ("PART VII: no feature weak on all three measures", len(weak) == 0),
    ("PART VII: sender account emptied ranks first by mutual information", mi_table["mutual_information"].idxmax() == "orig_emptied"),
    ("PART VII: PCA did not improve Logistic Regression", pca_test["LR with PCA"] < pca_test["LR without PCA"]),
    ("PART VIII: all four Set A models tied within the tolerance", len(tied) == 4),
    ("PART VIII: Set A train/validation gap at most 0.01", overfit["gap"].abs().max() <= 0.01),
    ("PART VIII: E2 within 0.01 of E1 (reconciliation removal barely matters)", abs(pick("E1") - pick("E2")) <= 0.01),
    ("PART VIII: E3 within 0.01 of E1 (receiving balances barely matter)", abs(pick("E1") - pick("E3")) <= 0.01),
    ("PART VIII: E5 clearly below E2", pick("E5") < pick("E2") - 0.05),
    ("PART VIII: E6 shared settings below E5", pick("E6") < pick("E5")),
    ("PART VIII: E4 random split below E1", pick("E4") < pick("E1")),
    ("PART VIII: tuned E6 model at least 10x the rule benchmark", best_row_d["pr_auc"] >= 10 * rule_prauc),
    ("PART IX: cost measure on validation favours the lowest threshold",
     SCORING["E6"]["threshold"] == min(CONFIG["thresholds"])),
    ("PART IX: E6 scores understate risk in the test period",
     SCORING["E6"]["p_cal"].mean() < test[TARGET].mean()),
    ("PART X: E1 relies on the account-emptying pattern (top SHAP feature)", sa.index[0] == "orig_emptied"),
    ("PART XI: night has the highest false positive rate by time of day", REPORT["BIAS_TOP_FPR_PERIOD"] == "Night"),
    # CHANGE (v2): more claims that the report now makes in words
    ("PART V: every money field is right-skewed (mean above median)",
     bool((dist["skew"] > 0).all() and (dist["mean"] > dist["median"]).all())),
    ("PART V: amount outliers have a higher fraud rate than overall",
     outlier_summary["fraud_rate_in_outliers_pct"].iloc[0] > outlier_summary["fraud_rate_overall_pct"].iloc[0]),
    ("PART VI: fraud rate at night is higher than the rest of the day",
     df_night_rate > df_day_rate),
    ("PART VI: legitimate daily volume is lower in the second half of the month",
     daily.loc[daily.index > half, "legit"].mean() < daily.loc[daily.index <= half, "legit"].mean()),
    ("PART VI: no raw field has |correlation| above 0.3 with isFraud",
     corr["isFraud"].drop("isFraud").abs().max() < 0.3),
    ("PART VI: sender balances before/after correlate above 0.9",
     corr.loc["oldbalanceOrg", "newbalanceOrig"] > 0.9),
    ("PART VIII: XGBoost tuned faster than Random Forest on Set A",
     TUNE_TIME["XGBoost"] < TUNE_TIME["RandomForest"]),
    ("PART X: E2 top SHAP feature is a sender balance field",
     SHAP_IMPORTANCE["E2"].index[0] in ["amt_to_orig_bal", "oldbalanceOrg", "newbalanceOrig"]),
    ("PART X: E5 top SHAP feature is the receiving balance after the transfer",
     SHAP_IMPORTANCE["E5"].index[0] == "newbalanceDest"),
]
claims = pd.DataFrame(CLAIMS, columns=["claim", "holds"]).set_index("claim")
save_table(claims, "17_report_claim_checks")
for c, ok in CLAIMS:
    print(("  OK    " if ok else "  FAIL  ") + c)
# NOTE TO SELF: on my v1 run all of these held. If any FAIL after a rerun,
# fix the matching sentence in the report by hand before submitting.

# 17.2 Save every number the report quotes
with open(Path(CONFIG["tables_dir"]) / "report_numbers.json", "w") as f:
    json.dump(REPORT, f, indent=2)
print(f"Saved {len(REPORT)} report numbers to report_numbers.json")
