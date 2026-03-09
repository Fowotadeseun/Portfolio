"""
Healthcare SQL Analytics — Hospital Readmission Pipeline
Author: Oluwaseun Daniel Fowotade
Description: End-to-end analytics pipeline using SQLite and Python to analyse
             hospital readmission patterns, patient demographics, and cost drivers.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#5C4033"]

import os
os.makedirs("images", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1. BUILD SQLite DATABASE
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)
N = 5000

conn = sqlite3.connect(":memory:")
cur  = conn.cursor()

# ── Patients table ────────────────────────────────────────────────
ages        = np.clip(np.random.normal(58, 18, N), 18, 95).astype(int)
sexes       = np.random.choice(["Male", "Female"], N)
races       = np.random.choice(["White", "Black", "Hispanic", "Asian", "Other"],
                                N, p=[0.60, 0.18, 0.13, 0.06, 0.03])
insurances  = np.random.choice(["Medicare", "Medicaid", "Private", "Uninsured"],
                                N, p=[0.38, 0.22, 0.32, 0.08])

cur.execute("""
    CREATE TABLE patients (
        patient_id   INTEGER PRIMARY KEY,
        age          INTEGER,
        sex          TEXT,
        race         TEXT,
        insurance    TEXT
    )
""")
for i in range(N):
    cur.execute("INSERT INTO patients VALUES (?,?,?,?,?)",
                (i + 1, int(ages[i]), sexes[i], races[i], insurances[i]))

# ── Admissions table ──────────────────────────────────────────────
diagnoses = np.random.choice(
    ["Heart Failure", "COPD", "Pneumonia", "Diabetes", "Sepsis",
     "Stroke", "Hip Fracture", "AMI"],
    N, p=[0.20, 0.15, 0.15, 0.18, 0.10, 0.08, 0.07, 0.07]
)
los         = np.clip(np.random.exponential(4.5, N), 1, 30).astype(int)
cost_base   = {"Heart Failure": 8000, "COPD": 6000, "Pneumonia": 7000,
               "Diabetes": 5000, "Sepsis": 14000, "Stroke": 10000,
               "Hip Fracture": 12000, "AMI": 11000}
costs       = np.array([cost_base[d] * (1 + np.random.uniform(-0.3, 0.5)) for d in diagnoses])
num_meds    = np.clip(np.random.poisson(8, N), 1, 25)
readmit_log = (-3.0
               + 0.025 * (ages - 58)
               + 0.40 * (insurances == "Medicaid")
               + 0.30 * (insurances == "Uninsured")
               + 0.35 * (diagnoses == "Heart Failure")
               + 0.28 * (diagnoses == "COPD")
               + 0.08 * los
               + 0.04 * num_meds
               + np.random.normal(0, 0.5, N))
readmit_prob    = 1 / (1 + np.exp(-readmit_log))
readmitted_30d  = (np.random.rand(N) < readmit_prob).astype(int)

cur.execute("""
    CREATE TABLE admissions (
        admission_id    INTEGER PRIMARY KEY,
        patient_id      INTEGER,
        diagnosis       TEXT,
        length_of_stay  INTEGER,
        total_cost      REAL,
        num_medications INTEGER,
        readmitted_30d  INTEGER,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
""")
for i in range(N):
    cur.execute("INSERT INTO admissions VALUES (?,?,?,?,?,?,?)",
                (i + 1, i + 1, diagnoses[i], int(los[i]),
                 round(costs[i], 2), int(num_meds[i]), int(readmitted_30d[i])))

conn.commit()

# ══════════════════════════════════════════════════════════════════
# 2. SQL QUERIES
# ══════════════════════════════════════════════════════════════════

Q1 = """
    SELECT
        a.diagnosis,
        COUNT(*)                         AS total_cases,
        ROUND(AVG(a.total_cost), 2)      AS avg_cost,
        ROUND(AVG(a.length_of_stay), 2)  AS avg_los,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM admissions a
    GROUP BY a.diagnosis
    ORDER BY readmit_rate_pct DESC
"""

Q2 = """
    SELECT
        p.insurance,
        COUNT(*)                        AS patients,
        ROUND(AVG(a.total_cost), 2)     AS avg_cost,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
    GROUP BY p.insurance
    ORDER BY avg_cost DESC
"""

Q3 = """
    SELECT
        CASE
            WHEN p.age BETWEEN 18 AND 40 THEN '18-40'
            WHEN p.age BETWEEN 41 AND 60 THEN '41-60'
            WHEN p.age BETWEEN 61 AND 80 THEN '61-80'
            ELSE '81+'
        END AS age_group,
        COUNT(*)                        AS patients,
        ROUND(AVG(a.length_of_stay), 2) AS avg_los,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
    GROUP BY age_group
    ORDER BY age_group
"""

Q4 = """
    SELECT
        p.race,
        COUNT(*)                        AS patients,
        ROUND(AVG(a.total_cost), 2)     AS avg_cost,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
    GROUP BY p.race
    ORDER BY readmit_rate_pct DESC
"""

df_diag   = pd.read_sql(Q1, conn)
df_ins    = pd.read_sql(Q2, conn)
df_age    = pd.read_sql(Q3, conn)
df_race   = pd.read_sql(Q4, conn)

print("Readmission by Diagnosis")
print(df_diag.to_string(index=False))
print("\nReadmission by Insurance")
print(df_ins.to_string(index=False))

# Save SQL queries to file for display
with open("queries.sql", "w") as f:
    f.write("-- Q1: Readmission rate and cost by diagnosis\n" + Q1 + "\n\n")
    f.write("-- Q2: Average cost and readmission by insurance type\n" + Q2 + "\n\n")
    f.write("-- Q3: Readmission rate by age group\n" + Q3 + "\n\n")
    f.write("-- Q4: Cost and readmission disparities by race/ethnicity\n" + Q4 + "\n")

# ══════════════════════════════════════════════════════════════════
# 3. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════

# Fig 1 — Readmission rate by diagnosis
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(df_diag["diagnosis"], df_diag["readmit_rate_pct"],
               color=PALETTE[0], edgecolor="white", height=0.6)
for bar in bars:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%", va="center", fontsize=10)
ax.set_xlabel("30-Day Readmission Rate (%)", fontsize=12)
ax.set_title("30-Day Hospital Readmission Rates by Diagnosis", fontsize=14, fontweight="bold")
ax.set_xlim(0, df_diag["readmit_rate_pct"].max() + 8)
plt.tight_layout()
plt.savefig("images/readmit_by_diagnosis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: readmit_by_diagnosis.png")

# Fig 2 — Average cost by insurance
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(df_ins["insurance"], df_ins["avg_cost"],
              color=PALETTE[:len(df_ins)], edgecolor="white", width=0.55)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"${bar.get_height():,.0f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Average Admission Cost (USD)", fontsize=12)
ax.set_title("Average Hospitalisation Cost by Insurance Type", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
plt.tight_layout()
plt.savefig("images/cost_by_insurance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cost_by_insurance.png")

# Fig 3 — Readmission rate by age group
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_age["age_group"], df_age["readmit_rate_pct"],
       color=PALETTE, edgecolor="white", width=0.55)
for i, (rate, los) in enumerate(zip(df_age["readmit_rate_pct"], df_age["avg_los"])):
    ax.text(i, rate + 0.3, f"{rate:.1f}%\n(LOS {los:.1f}d)",
            ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Age Group", fontsize=12)
ax.set_ylabel("30-Day Readmission Rate (%)", fontsize=12)
ax.set_title("Readmission Rate and Average LOS by Age Group", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("images/readmit_by_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: readmit_by_age.png")

# Fig 4 — Cost distribution by diagnosis (boxplot)
df_full = pd.read_sql(
    "SELECT a.diagnosis, a.total_cost FROM admissions a", conn
)
fig, ax = plt.subplots(figsize=(12, 5))
order = df_full.groupby("diagnosis")["total_cost"].median().sort_values(ascending=False).index
sns.boxplot(data=df_full, x="diagnosis", y="total_cost", order=order,
            palette=PALETTE * 2, ax=ax, width=0.55, linewidth=1.5)
ax.set_xlabel("")
ax.set_ylabel("Total Cost (USD)", fontsize=12)
ax.set_title("Hospitalisation Cost Distribution by Diagnosis", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig("images/cost_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cost_distribution.png")

# Fig 5 — Readmission disparity by race
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_race, x="race", y="readmit_rate_pct",
            palette=PALETTE, ax=ax, edgecolor="white")
ax.set_xlabel("")
ax.set_ylabel("30-Day Readmission Rate (%)", fontsize=12)
ax.set_title("Readmission Disparities by Race / Ethnicity", fontsize=14, fontweight="bold")
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.3,
            f"{p.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("images/readmit_by_race.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: readmit_by_race.png")

# ══════════════════════════════════════════════════════════════════
# 4. PREDICTIVE MODEL — 30-day Readmission
# ══════════════════════════════════════════════════════════════════
df_model = pd.read_sql("""
    SELECT p.age, p.sex, p.race, p.insurance,
           a.diagnosis, a.length_of_stay,
           a.num_medications, a.total_cost, a.readmitted_30d
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
""", conn)

le = LabelEncoder()
for col in ["sex", "race", "insurance", "diagnosis"]:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("readmitted_30d", axis=1)
y = df_model["readmitted_30d"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_tr)
proba = lr.predict_proba(X_te_s)[:, 1]
auc   = roc_auc_score(y_te, proba)
print(f"\nLogistic Regression AUC: {auc:.3f}")
print(classification_report(y_te, lr.predict(X_te_s)))

# Feature importance
coef_df = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": lr.coef_[0]
}).sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(9, 5))
colors_c = [PALETTE[1] if c > 0 else PALETTE[0] for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"],
        color=colors_c, edgecolor="white", height=0.65)
ax.axvline(0, color="black", lw=1.2)
ax.set_xlabel("Logistic Regression Coefficient", fontsize=12)
ax.set_title("Predictors of 30-Day Hospital Readmission", fontsize=14, fontweight="bold")
red_patch  = mpatches.Patch(color=PALETTE[1], label="Increases readmission risk")
blue_patch = mpatches.Patch(color=PALETTE[0], label="Decreases readmission risk")
ax.legend(handles=[red_patch, blue_patch], fontsize=10)
plt.tight_layout()
plt.savefig("images/readmit_predictors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: readmit_predictors.png")

conn.close()
print("\nAll figures saved to images/")

