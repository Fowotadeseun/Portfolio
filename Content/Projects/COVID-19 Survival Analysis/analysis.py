"""
COVID-19 Clinical Survival Analysis
Author: Oluwaseun Daniel Fowotade
Description: Kaplan-Meier survival analysis and Cox proportional hazards modeling
             on synthetic COVID-19 hospitalization data to investigate mortality risk factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#5C4033"]

import os
os.makedirs("images", exist_ok=True)

# ── 1. Generate Synthetic COVID-19 Clinical Dataset ──────────────────────────
np.random.seed(2024)
n = 1200

age      = np.clip(np.random.normal(58, 18, n), 18, 95).astype(int)
sex      = np.random.binomial(1, 0.55, n)          # 1=Male
icu      = np.random.binomial(1, 0.28, n)
ventilated = (icu & (np.random.rand(n) < 0.45)).astype(int)
comorbidity = np.random.binomial(1, 0.60, n)       # hypertension/diabetes/etc.
severity    = np.random.choice(["Mild", "Moderate", "Severe"], n,
                                p=[0.35, 0.40, 0.25])
severity_num = np.where(severity == "Mild", 0, np.where(severity == "Moderate", 1, 2))

# Hazard: older, male, ICU, severe → higher mortality risk
log_hazard = (
    -6.5
    + 0.04  * (age - 58)
    + 0.30  * sex
    + 1.20  * icu
    + 1.80  * ventilated
    + 0.40  * comorbidity
    + 0.70  * severity_num
    + np.random.normal(0, 0.4, n)
)
hazard   = np.exp(log_hazard)
# Exponential event times
time_to_event = np.random.exponential(1 / (hazard + 1e-6))
# Censor at 60 days
obs_time = np.minimum(time_to_event, 60)
event    = (time_to_event <= 60).astype(int)

df = pd.DataFrame({
    "PatientID":   np.arange(1, n + 1),
    "Age":         age,
    "Sex":         np.where(sex == 1, "Male", "Female"),
    "ICU":         icu,
    "Ventilated":  ventilated,
    "Comorbidity": comorbidity,
    "Severity":    severity,
    "ObsTime":     np.round(obs_time, 2),
    "Event":       event,           # 1 = died
})

print(f"Dataset: {n} patients, {event.sum()} deaths ({event.mean():.1%} mortality)")
print(df.head())

# ── 2. Kaplan-Meier Estimator (manual) ───────────────────────────────────────
def kaplan_meier(times, events):
    """Compute KM survival estimate."""
    df_km = pd.DataFrame({"t": times, "e": events}).sort_values("t")
    unique_t = sorted(df_km["t"][df_km["e"] == 1].unique())
    S = 1.0
    records = [(0, 1.0)]
    n_at_risk = len(df_km)
    for t in unique_t:
        d = df_km[(df_km["t"] == t) & (df_km["e"] == 1)].shape[0]
        S = S * (1 - d / n_at_risk)
        records.append((t, S))
        n_at_risk -= df_km[df_km["t"] == t].shape[0]
    return pd.DataFrame(records, columns=["time", "survival"])

# ── 3. Overall KM Curve ───────────────────────────────────────────────────────
km_all = kaplan_meier(df["ObsTime"], df["Event"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.step(km_all["time"], km_all["survival"], where="post",
        color=PALETTE[0], lw=2.5, label="Overall (n=1,200)")
ax.fill_between(km_all["time"], km_all["survival"] * 0.95,
                km_all["survival"] * 1.05,
                alpha=0.15, color=PALETTE[0], step="post")
ax.axhline(0.5, ls="--", color="grey", lw=1, label="50% survival")
ax.set_xlabel("Days from Admission", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.set_title("Kaplan–Meier Survival Curve — All Patients", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("images/km_overall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: km_overall.png")

# ── 4. KM by Severity ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for sev, color in zip(["Mild", "Moderate", "Severe"], PALETTE):
    sub = df[df["Severity"] == sev]
    km  = kaplan_meier(sub["ObsTime"], sub["Event"])
    ax.step(km["time"], km["survival"], where="post", color=color,
            lw=2.5, label=f"{sev} (n={len(sub)})")
ax.set_xlabel("Days from Admission", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.set_title("Kaplan–Meier Curves by Disease Severity", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, title="Severity", title_fontsize=11)
ax.axhline(0.5, ls="--", color="grey", lw=1)
plt.tight_layout()
plt.savefig("images/km_by_severity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: km_by_severity.png")

# ── 5. KM by ICU Status ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for icu_val, label, color in zip([0, 1], ["Non-ICU", "ICU"], PALETTE):
    sub = df[df["ICU"] == icu_val]
    km  = kaplan_meier(sub["ObsTime"], sub["Event"])
    ax.step(km["time"], km["survival"], where="post", color=color,
            lw=2.5, label=f"{label} (n={len(sub)})")
ax.set_xlabel("Days from Admission", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.set_title("Kaplan–Meier Curves by ICU Admission", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.axhline(0.5, ls="--", color="grey", lw=1)
plt.tight_layout()
plt.savefig("images/km_icu.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: km_icu.png")

# ── 6. KM by Age Group ────────────────────────────────────────────────────────
df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 40, 60, 80, 100],
                         labels=["18–40", "41–60", "61–80", "81+"])

fig, ax = plt.subplots(figsize=(10, 6))
for grp, color in zip(["18–40", "41–60", "61–80", "81+"], PALETTE):
    sub = df[df["AgeGroup"] == grp]
    km  = kaplan_meier(sub["ObsTime"], sub["Event"])
    ax.step(km["time"], km["survival"], where="post", color=color,
            lw=2.5, label=f"{grp} (n={len(sub)})")
ax.set_xlabel("Days from Admission", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.set_title("Kaplan–Meier Curves by Age Group", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, title="Age Group")
ax.axhline(0.5, ls="--", color="grey", lw=1)
plt.tight_layout()
plt.savefig("images/km_age_groups.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: km_age_groups.png")

# ── 7. Cox PH — Manual log-rank & HR table ────────────────────────────────────
def log_rank_test(t1, e1, t2, e2):
    """Simplified log-rank test statistic."""
    all_t = sorted(set(t1[e1 == 1]) | set(t2[e2 == 1]))
    O1 = E1 = O2 = E2 = 0
    n1_r = len(t1)
    n2_r = len(t2)
    for t in all_t:
        n_r = n1_r + n2_r
        if n_r == 0:
            break
        d1 = ((t1 == t) & (e1 == 1)).sum()
        d2 = ((t2 == t) & (e2 == 1)).sum()
        d  = d1 + d2
        e1_exp = d * n1_r / n_r
        e2_exp = d * n2_r / n_r
        O1 += d1; E1 += e1_exp
        O2 += d2; E2 += e2_exp
        n1_r -= ((t1 == t)).sum()
        n2_r -= ((t2 == t)).sum()
    chi2 = (O1 - E1) ** 2 / (E1 + 1e-9) + (O2 - E2) ** 2 / (E2 + 1e-9)
    p    = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p

# Hazard ratio approximation via mortality rates
hr_table = []
comparisons = [
    ("Sex",         "Male",   "Female"),
    ("ICU",         1,        0),
    ("Ventilated",  1,        0),
    ("Comorbidity", 1,        0),
]
for col, case, ctrl in comparisons:
    g_case = df[df[col] == case]
    g_ctrl = df[df[col] == ctrl]
    rate_case = g_case["Event"].sum() / g_case["ObsTime"].sum()
    rate_ctrl = g_ctrl["Event"].sum() / g_ctrl["ObsTime"].sum()
    hr = rate_case / (rate_ctrl + 1e-9)
    _, p = log_rank_test(g_case["ObsTime"].values, g_case["Event"].values,
                         g_ctrl["ObsTime"].values, g_ctrl["Event"].values)
    hr_table.append({"Factor": f"{col}: {case} vs {ctrl}",
                     "HR": round(hr, 2),
                     "p-value": round(p, 4)})

hr_df = pd.DataFrame(hr_table)
print("\nHazard Ratios:")
print(hr_df.to_string(index=False))

# ── 8. Forest Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(hr_df))
colors_fp = [PALETTE[1] if hr > 1 else PALETTE[0] for hr in hr_df["HR"]]
ax.barh(y_pos, hr_df["HR"] - 1, left=1, color=colors_fp, alpha=0.8,
        edgecolor="white", height=0.55)
ax.axvline(1.0, color="black", lw=1.5, ls="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(hr_df["Factor"], fontsize=11)
ax.set_xlabel("Hazard Ratio", fontsize=12)
ax.set_title("Estimated Hazard Ratios for COVID-19 Mortality", fontsize=14, fontweight="bold")
for i, (hr, p) in enumerate(zip(hr_df["HR"], hr_df["p-value"])):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.text(hr + 0.05, i, f"HR={hr:.2f} {sig}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("images/forest_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: forest_plot.png")

# ── 9. Mortality by Severity × ICU ────────────────────────────────────────────
summary = df.groupby(["Severity", "ICU"])["Event"].mean().reset_index()
summary["ICU_label"] = summary["ICU"].map({0: "Non-ICU", 1: "ICU"})
summary["MortalityPct"] = summary["Event"] * 100

fig, ax = plt.subplots(figsize=(8, 5))
order   = ["Mild", "Moderate", "Severe"]
x       = np.arange(len(order))
w       = 0.35
for i, (icu_label, color) in enumerate(zip(["Non-ICU", "ICU"], PALETTE)):
    vals = [summary[(summary["Severity"] == s) & (summary["ICU_label"] == icu_label)]["MortalityPct"].values
            for s in order]
    vals = [v[0] if len(v) > 0 else 0 for v in vals]
    rects = ax.bar(x + i * w - w / 2, vals, w, label=icu_label,
                   color=color, alpha=0.85, edgecolor="white")
    for r in rects:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5,
                f"{r.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=12)
ax.set_ylabel("30-Day Mortality (%)", fontsize=12)
ax.set_title("Mortality Rate by Severity and ICU Admission", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("images/mortality_severity_icu.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: mortality_severity_icu.png")

print("\nAll figures saved to images/")
