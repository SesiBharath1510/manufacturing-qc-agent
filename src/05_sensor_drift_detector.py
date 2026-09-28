import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f

# 1. Simulate 500 timestamps of assembly line telemetry
np.random.seed(42)
n_samples = 500

# Normal baseline operating parameters:
# - Spindle Temp: 65.0 °C ± 2.0
# - Vibration Amplitude: 1.2 mm/s ± 0.15
# - Spindle RPM: 3000 ± 25
timestamps = pd.date_range("2026-03-29 08:00:00", periods=n_samples, freq="10s")

temp = np.random.normal(loc=65.0, scale=2.0, size=n_samples)
vibration = np.random.normal(loc=1.2, scale=0.15, size=n_samples)
rpm = np.random.normal(loc=3000.0, scale=25.0, size=n_samples)

# Inject drift starting at sample index 350 (bearing degradation)
# Temp slowly climbs, vibration increases, RPM fluctuates
for i in range(350, n_samples):
    drift_factor = (i - 350) * 0.08
    temp[i] += drift_factor * 1.5
    vibration[i] += drift_factor * 0.06
    rpm[i] -= drift_factor * 2.0

df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_c": temp,
    "vibration_mms": vibration,
    "rpm": rpm
})

# Save raw simulated telemetry to CSV
os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "machine_telemetry.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Telemetry data generated and saved to: {csv_path}")

# =========================================================
# 2. Statistical Technique 1: EWMA on Temperature
# =========================================================
lam = 0.2  # Smoothing parameter (weight on recent observation)
baseline_mean = df["temperature_c"][:300].mean()
baseline_std = df["temperature_c"][:300].std()

# Upper Control Limit (UCL) for EWMA: 3 sigma boundary
ucl_ewma = baseline_mean + 3 * baseline_std * np.sqrt(lam / (2 - lam))

ewma_values = []
current_z = baseline_mean
for val in df["temperature_c"]:
    current_z = lam * val + (1 - lam) * current_z
    ewma_values.append(current_z)

df["ewma_temp"] = ewma_values
df["ewma_alert"] = df["ewma_temp"] > ucl_ewma

# =========================================================
# 3. Statistical Technique 2: Hotelling's T² (MSPC)
# =========================================================
features = ["temperature_c", "vibration_mms", "rpm"]
X = df[features].values

# Baseline healthy data (first 300 samples)
X_baseline = X[:300]
mu_vec = np.mean(X_baseline, axis=0)
cov_matrix = np.cov(X_baseline, rowvar=False)
inv_cov = np.linalg.inv(cov_matrix)

# Compute T² for every observation: (x - μ)ᵀ Σ⁻¹ (x - μ)
t2_values = []
for row in X:
    diff = row - mu_vec
    t2 = np.dot(np.dot(diff, inv_cov), diff.T)
    t2_values.append(t2)

df["hotelling_t2"] = t2_values

# Calculate Upper Control Limit using F-distribution critical threshold
p = len(features)  # 3 variables
n = len(X_baseline)  # 300 baseline observations
alpha = 0.01  # 99% confidence level
f_crit = f.ppf(1 - alpha, dfn=p, dfd=n - p)
ucl_t2 = (p * (n - 1) / (n - p)) * f_crit

df["mspc_alert"] = df["hotelling_t2"] > ucl_t2

# Print summary
first_ewma_breach = df[df["ewma_alert"]].index.min()
first_t2_breach = df[df["mspc_alert"]].index.min()

print("\n--- STATISTICAL PROCESS CONTROL SUMMARY ---")
print(f"Baseline Temperature Mean: {baseline_mean:.2f} °C (EWMA UCL: {ucl_ewma:.2f} °C)")
print(f"Hotelling T² 99% Control Limit (UCL): {ucl_t2:.2f}")
print(f"First EWMA breach detected at step: {first_ewma_breach} (Time: {df.loc[first_ewma_breach, 'timestamp']})")
print(f"First Hotelling T² breach detected at step: {first_t2_breach} (Time: {df.loc[first_t2_breach, 'timestamp']})")

# =========================================================
# 4. Generate & Save SPC Control Chart Plot
# =========================================================
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df.index, df["temperature_c"], label="Raw Temp (°C)", color="lightgray")
plt.plot(df.index, df["ewma_temp"], label="EWMA Smoothed Temp", color="blue")
plt.axhline(ucl_ewma, color="red", linestyle="--", label=f"EWMA UCL ({ucl_ewma:.1f} °C)")
plt.axvline(350, color="orange", linestyle=":", label="Drift Onset (Step 350)")
plt.title("Univariate Drift: EWMA Chart (Spindle Temperature)")
plt.legend(loc="upper left")

plt.subplot(2, 1, 2)
plt.plot(df.index, df["hotelling_t2"], label="Hotelling T² Metric", color="purple")
plt.axhline(ucl_t2, color="red", linestyle="--", label=f"T² UCL ({ucl_t2:.1f})")
plt.axvline(350, color="orange", linestyle=":", label="Drift Onset (Step 350)")
plt.title("Multivariate Drift: Hotelling's T² Chart (Temp + Vibration + RPM)")
plt.xlabel("Sample Index")
plt.legend(loc="upper left")

plt.tight_layout()
chart_path = os.path.join("output", "spc_drift_charts.png")
plt.savefig(chart_path)
print(f"✅ Control chart visualizations saved to: {chart_path}")