"""
generate_dataset.py — Creates a synthetic METR-LA-like dataset (metr_la_synthetic.csv)
Run once before starting the backend: python generate_dataset.py

Produces 207 sensors x 12,000 timestamps (1000 hours of 5-min data) of realistic
traffic speeds with morning/evening rush, incidents, weekend effects, and spatial correlation.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SENSORS   = 207
N_STEPS     = 12000          # ~1000 hours at 5-min intervals
INTERVAL    = 5              # minutes
FREEWAY_SPD = 65.0           # free-flow speed (mph)
OUT_PATH    = os.path.join(os.path.dirname(__file__), "metr_la_synthetic.csv")
ADJ_PATH    = os.path.join(os.path.dirname(__file__), "adj_synthetic.npy")

print(f"[generate] Building {N_SENSORS} sensors × {N_STEPS} steps …")

# ── 1. Time grid ───────────────────────────────────────────────
timestamps = pd.date_range("2012-03-01 00:00", periods=N_STEPS, freq=f"{INTERVAL}min")
hour_of_day = timestamps.hour + timestamps.minute / 60.0
day_of_week = timestamps.dayofweek                         # 0=Mon … 6=Sun

# ── 2. Base demand profile (shared across sensors) ────────────
def demand_profile(hour, dow):
    """Returns speed reduction factor ∈ [0,1] (0=free-flow, 1=severe congestion)."""
    is_weekend = dow >= 5
    # Morning peak 7–9 am, evening peak 4–7 pm (weekday)
    am_peak = np.exp(-0.5 * ((hour - 8.0) / 0.8) ** 2)
    pm_peak = np.exp(-0.5 * ((hour - 17.5) / 1.2) ** 2)
    weekday = 0.35 * am_peak + 0.45 * pm_peak
    weekend = 0.15 * np.exp(-0.5 * ((hour - 12.5) / 2.0) ** 2)
    return np.where(is_weekend, weekend, weekday)

base_demand = demand_profile(hour_of_day.values, day_of_week.values)   # shape (N_STEPS,)

# ── 3. Per-sensor heterogeneity ────────────────────────────────
# Sensors on different road types / bottlenecks
sensor_severity = np.random.beta(2, 5, N_SENSORS)          # how badly each sensor congests
sensor_freeflow = np.random.normal(FREEWAY_SPD, 4, N_SENSORS).clip(50, 75)

# ── 4. Spatial adjacency (ring + random long-range) ───────────
adj = np.zeros((N_SENSORS, N_SENSORS), dtype=np.float32)
for i in range(N_SENSORS):
    for d in [1, 2, 3]:
        j = (i + d) % N_SENSORS
        w = 1.0 / d
        adj[i, j] = w; adj[j, i] = w
# Add ~3 random long-range links per sensor
for i in range(N_SENSORS):
    rng_links = np.random.choice(N_SENSORS, 3, replace=False)
    for j in rng_links:
        if j != i:
            adj[i, j] = 0.3; adj[j, i] = 0.3
np.save(ADJ_PATH, adj)

# ── 5. Build speed matrix ─────────────────────────────────────
speeds = np.zeros((N_STEPS, N_SENSORS), dtype=np.float32)
incident_log = []

for i in range(N_SENSORS):
    ff   = sensor_severity[i]
    v_ff = sensor_freeflow[i]

    # Base speed from demand
    v_base = v_ff * (1.0 - ff * base_demand)

    # Spatially correlated noise (smoothed Gaussian)
    noise = np.random.randn(N_STEPS).cumsum()
    noise -= noise.mean()
    noise /= (noise.std() + 1e-6)
    noise *= 2.0                                           # ±2 mph noise

    v = v_base + noise

    # Inject 3–7 random non-recurrent incidents per sensor
    n_incidents = np.random.randint(3, 8)
    for _ in range(n_incidents):
        t0  = np.random.randint(0, N_STEPS - 60)
        dur = np.random.randint(12, 48)                    # 1–4 hrs
        drop = np.random.uniform(15, 40)                   # speed drop (mph)
        ramp_up = np.linspace(0, drop, min(6, dur // 2))
        ramp_dn = np.linspace(drop, 0, dur - len(ramp_up))
        profile = np.concatenate([ramp_up, ramp_dn])
        end = min(t0 + len(profile), N_STEPS)
        v[t0:end] -= profile[:end - t0]
        incident_log.append((i, t0, end))

    speeds[:, i] = np.clip(v, 5.0, v_ff)

# ── 6. Propagate incidents spatially ──────────────────────────
for (i, t0, t1) in incident_log:
    for j in range(N_SENSORS):
        w = adj[i, j]
        if w > 0.3:
            delay = np.random.randint(1, 4)
            t0d = min(t0 + delay, N_STEPS - 1)
            t1d = min(t1 + delay, N_STEPS)
            speeds[t0d:t1d, j] = np.maximum(
                speeds[t0d:t1d, j] - w * np.random.uniform(5, 20),
                5.0
            )

# ── 7. Save CSV ───────────────────────────────────────────────
cols = [f"sensor_{i:03d}" for i in range(N_SENSORS)]
df   = pd.DataFrame(speeds, index=timestamps, columns=cols)
df.index.name = "timestamp"
df.to_csv(OUT_PATH)
print(f"[generate] Saved {OUT_PATH}  shape={df.shape}")
print(f"[generate] Speed stats: min={speeds.min():.1f}  mean={speeds.mean():.1f}  max={speeds.max():.1f} mph")
print(f"[generate] Total incidents injected: {len(incident_log)}")
print("[generate] Done.")
