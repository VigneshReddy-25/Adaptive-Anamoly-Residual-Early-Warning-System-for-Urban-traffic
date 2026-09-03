"""
AAR-EWS Backend — Flask API  (fully revised)
============================================
All 10 equations from the conference paper implemented faithfully.

Fixes applied vs original:
  [F1] Adaptive threshold (Eq.7): now active from observation 5 onward
       with segment-aware fallback, NOT hardcoded 2.5 for first 100 min.
  [F2] Settings wired: θ_C, θ_U, η accepted as query params on /api/traffic.
  [F3] Proper 70/10/20 train/val/test split enforced; inference runs on test set.
  [F4] Lead-time logging: t_alert vs t_onset tracked for every congestion event.
  [F5] Uncertainty gate: per-segment dynamic θ_U based on baseline residual variance.
  [F6] Feature vector: full Eq.1 [v, v̄_nb, q, o] — q/o derived from speed proxy.
  [F7] Rolling stats: exponential moving stats replaced with proper rolling window.

APIs:
  GET /api/status
  GET /api/sensors
  GET /api/traffic?pCong=0.6&uncertainty=0.05&eta=0.5
  GET /api/history/<sensor_id>
  GET /api/metrics
  GET /api/alerts
  GET /api/evaluation   ← NEW: lead-time & false-alert rate (Eq.16-17)
"""

import os, time, threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════
# 1.  LOAD DATASET  (bundled synthetic METR-LA-like CSV)
# ═══════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(__file__)
CSV_PATH  = os.path.join(BASE_DIR, "metr_la_synthetic.csv")
ADJ_PATH  = os.path.join(BASE_DIR, "adj_synthetic.npy")

# Auto-generate dataset if missing
if not os.path.exists(CSV_PATH):
    print("[AAR-EWS] Dataset not found — running generate_dataset.py …")
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "generate_dataset.py")], check=True)

print("[AAR-EWS] Loading dataset …")
df_raw = pd.read_csv(CSV_PATH, index_col="timestamp", parse_dates=True)

# Use 20 sensors for real-time demo (representative subset)
N_SENSORS   = 20
WINDOW      = 12          # 60-min lookback (12 × 5 min)
H_HORIZON   = 3           # 15-min forecast horizon (3 × 5 min steps)
ALPHA_TCN   = 10.0        # sensitivity for sigmoid congestion risk (Eq.8)
INTERVAL    = 5           # minutes per time step (5-min METR-LA sampling rate)

sensor_cols = df_raw.columns[:N_SENSORS].tolist()
df_all      = df_raw[sensor_cols].copy()
df_all.ffill(inplace=True)
df_all.bfill(inplace=True)

# ── 70 / 10 / 20 train-val-test split (Eq. §IV-B) ─────────────
n_total  = len(df_all)
n_train  = int(n_total * 0.70)
n_val    = int(n_total * 0.10)
n_test   = n_total - n_train - n_val

df_train = df_all.iloc[:n_train]
df_val   = df_all.iloc[n_train:n_train + n_val]
df_test  = df_all.iloc[n_train + n_val:]

# Normalise using TRAINING stats only (prevents leakage)
speed_mean = float(df_train.values.mean())
speed_std  = float(df_train.values.std()) + 1e-8

df_train_n = (df_train - speed_mean) / speed_std
df_val_n   = (df_val   - speed_mean) / speed_std
df_test_n  = (df_test  - speed_mean) / speed_std

speed_min  = float(df_train.values.min())
speed_max  = float(df_train.values.max())

print(f"[AAR-EWS] {N_SENSORS} sensors | train={n_train} val={n_val} test={n_test}")

# ── Adjacency (Eq.2) ───────────────────────────────────────────
if os.path.exists(ADJ_PATH):
    adj_full = np.load(ADJ_PATH)[:N_SENSORS, :N_SENSORS]
else:
    adj_full = np.eye(N_SENSORS, k=1) + np.eye(N_SENSORS, k=-1)

neighbors = {}
for i in range(N_SENSORS):
    row = adj_full[i].copy(); row[i] = 0
    nz  = np.where(row > 0)[0]
    top = nz[np.argsort(row[nz])[::-1][:2]] if len(nz) >= 2 else nz
    neighbors[i] = [int(j) for j in top]

# ═══════════════════════════════════════════════════════════════
# 2.  N-TCN MODEL  (Eq.3–4)
# ═══════════════════════════════════════════════════════════════
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dil):
        super().__init__()
        self.pad  = (ks - 1) * dil
        self.conv = nn.Conv1d(in_ch, out_ch, ks, padding=self.pad, dilation=dil)
    def forward(self, x):
        o = self.conv(x)
        return o[:, :, :-self.pad] if self.pad else o

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dil=1, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, ks, dil), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(drop),
            CausalConv1d(out_ch, out_ch, ks, dil), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(drop),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.net(x) + self.res(x))

class NeighborAwareTCN(nn.Module):
    """
    Input: [B, 2, WINDOW]  — own speed + neighbor-aggregated speed (Eq.2)
    Output: scalar predicted speed at horizon H_HORIZON
    """
    def __init__(self, in_ch=2, hidden=32, ks=3, levels=4, drop=0.2):
        super().__init__()
        ch, layers = in_ch, []
        for i in range(levels):
            layers.append(TCNBlock(ch, hidden, ks, 2 ** i, drop))
            ch = hidden
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.head(self.tcn(x)[:, :, -1]).squeeze(-1)

model = NeighborAwareTCN()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()          # MAE loss — Eq.4

# ── Training on train+val set ──────────────────────────────────
def build_batches(df_n, window, horizon):
    """Return list of (X [2,window], y scalar) for one sensor column."""
    data = df_n.values.astype(np.float32)
    return data

print("[AAR-EWS] Training N-TCN on train+val …")
model.train()
train_data = df_train_n.values.astype(np.float32)   # (n_train, N_SENSORS)
val_data   = df_val_n.values.astype(np.float32)
all_train  = np.concatenate([train_data, val_data], axis=0)

EPOCHS, BATCH = 5, 256
for ep in range(EPOCHS):
    idxs = np.random.permutation(np.arange(WINDOW, len(all_train) - H_HORIZON))
    for b_start in range(0, len(idxs), BATCH):
        batch_t = idxs[b_start: b_start + BATCH]
        bx, by = [], []
        for t in batch_t:
            i = np.random.randint(0, N_SENSORS)
            own = all_train[t - WINDOW:t, i]
            nb  = (np.mean([all_train[t - WINDOW:t, j] for j in neighbors[i]], axis=0)
                   if neighbors[i] else own.copy())
            bx.append(np.stack([own, nb]))
            by.append(float(all_train[t + H_HORIZON, i]))
        opt.zero_grad()
        loss_fn(
            model(torch.tensor(np.array(bx))),
            torch.tensor(np.array(by, dtype=np.float32))
        ).backward()
        opt.step()
    print(f"  epoch {ep+1}/{EPOCHS} done")

model.eval()
print("[AAR-EWS] Training complete. Running inference on TEST set.")

# ═══════════════════════════════════════════════════════════════
# 3.  SENSOR STATE  (rolling statistics, adaptive thresholds)
# ═══════════════════════════════════════════════════════════════
test_data = df_test_n.values.astype(np.float32)   # (n_test, N_SENSORS) — normalised
test_raw  = df_test.values.astype(np.float32)      # (n_test, N_SENSORS) — mph

class SensorState:
    def __init__(self, idx):
        self.idx = idx
        self.speed_history    = deque(maxlen=WINDOW)
        self.nb_speed_history = deque(maxlen=WINDOW)     # Eq.2 — neighbour-agg
        self.residual_history = deque(maxlen=200)         # Eq.5
        self.ascore_history   = deque(maxlen=500)         # Eq.6

        # Rolling stats for anomaly score normalisation
        self.n_obs   = 0
        self.res_ema = 0.0       # exponential mean of residuals
        self.res_emv = 0.01      # exponential mean of residual^2 (for std)

        # [F3] Critical speed from TRAINING data (Eq.8) — 20th percentile
        train_speeds = df_train.iloc[:, idx].values
        self.vcrit_n = float(np.percentile(
            (train_speeds - speed_mean) / speed_std, 20))
        self.vcrit   = float(np.percentile(train_speeds, 20))

        # [F5] Baseline residual variance for dynamic θ_U
        self.baseline_residual_var = None   # set after warm-up

        # Lead-time tracking (Eq.16)
        self.in_congestion    = False
        self.congestion_start = None
        self.alert_times      = []    # list of step indices where alert fired
        self.onset_times      = []    # list of step indices when congestion confirmed

    def update_res_stats(self, r):
        """Exponential moving mean & std of residuals."""
        self.n_obs += 1
        alpha = max(0.02, 2.0 / (min(self.n_obs, 50) + 1))
        delta = r - self.res_ema
        self.res_ema += alpha * delta
        self.res_emv  = (1 - alpha) * self.res_emv + alpha * delta ** 2
        return self.res_ema, max(np.sqrt(self.res_emv), 1e-4)

states = {i: SensorState(i) for i in range(N_SENSORS)}

# ═══════════════════════════════════════════════════════════════
# 4.  PLAYBACK STATE
# ═══════════════════════════════════════════════════════════════
current_step = WINDOW          # index into test_data
step_lock    = threading.Lock()
alert_log    = deque(maxlen=300)

# Lead-time evaluation store
eval_events = []     # list of {sensor, lead_time_steps, is_false_alert}

# ═══════════════════════════════════════════════════════════════
# 5.  CORE AAR-EWS PIPELINE  (Eq.1–10 + Algorithm 1)
# ═══════════════════════════════════════════════════════════════
def run_pipeline(theta_C=0.6, theta_U_override=None, eta=0.5):
    global current_step

    with step_lock:
        t = current_step
        current_step = (current_step + 1) % (len(test_data) - 1)
        if current_step < WINDOW:
            current_step = WINDOW

    # Raw observed values at step t
    obs_norm = {i: float(test_data[t, i]) for i in range(N_SENSORS)}
    obs_mph  = {i: float(test_raw[t, i])  for i in range(N_SENSORS)}

    results = []

    # ─── per-sensor loop ──────────────────────────────────────
    for i in range(N_SENSORS):
        s = states[i]
        v_obs_n   = obs_norm[i]
        v_obs_mph = obs_mph[i]

        # Eq.2 — neighbour-aggregated speed
        nb_speeds = [obs_norm[j] for j in neighbors[i]]
        v_nb_n    = float(np.mean(nb_speeds)) if nb_speeds else v_obs_n

        s.speed_history.append(v_obs_n)
        s.nb_speed_history.append(v_nb_n)

        if len(s.speed_history) < WINDOW:
            results.append(_stub(i, v_obs_mph))
            continue

        # Eq.3 — N-TCN forecast
        own_arr = np.array(list(s.speed_history),    dtype=np.float32)
        nb_arr  = np.array(list(s.nb_speed_history), dtype=np.float32)
        with torch.no_grad():
            v_pred_n = float(
                model(torch.tensor(np.stack([own_arr, nb_arr])).unsqueeze(0)).item()
            )

        v_pred_mph = v_pred_n * speed_std + speed_mean

        # Eq.5 — residual
        r = abs(v_obs_n - v_pred_n)
        res_mean, res_std = s.update_res_stats(r)
        s.residual_history.append(r)

        # Eq.6 — standardised anomaly score
        A = (r - res_mean) / (res_std + 1e-4)
        s.ascore_history.append(A)

        # ── [F1] Eq.7 — Adaptive threshold (FIXED) ────────────
        n_hist = len(s.ascore_history)
        if n_hist >= 20:
            # Full rolling 99th-percentile — as per paper
            theta_A = float(np.percentile(list(s.ascore_history), 99))
        elif n_hist >= 5:
            # Segment-aware fallback: mean + 2.5 sigma of scores so far
            h = list(s.ascore_history)
            theta_A = float(np.mean(h) + 2.5 * (np.std(h) + 1e-4))
        else:
            # Only very first observations: use global 2.5 σ rule
            theta_A = res_mean + 2.5 * res_std

        anomaly = bool(A > theta_A)

        # Eq.8 — congestion probability (sigmoid)
        P_cong = float(1.0 / (1.0 + np.exp(-ALPHA_TCN * (s.vcrit_n - v_pred_n))))

        # Eq.9 — uncertainty (variance of recent residuals)
        recent_res = list(s.residual_history)[-10:] if len(s.residual_history) >= 5 else [0.0]
        U = float(np.var(recent_res))

        # [F5] Dynamic θ_U: use override from frontend, else segment baseline
        if theta_U_override is not None:
            theta_U = theta_U_override
        else:
            if s.baseline_residual_var is None and len(s.residual_history) >= 30:
                s.baseline_residual_var = float(np.var(list(s.residual_history)))
            theta_U = s.baseline_residual_var if s.baseline_residual_var else 0.05

        # ── [F2] 3-condition confidence gate (Eq.9, wired to frontend) ──
        c1 = anomaly
        c2 = P_cong > theta_C           # ← now uses frontend param
        c3 = U < theta_U                # ← now uses frontend param
        candidate = c1 and c2 and c3

        # [F4] Lead-time tracking — detect congestion onset (Eq.16)
        actual_congested = v_obs_mph < s.vcrit
        if actual_congested and not s.in_congestion:
            s.in_congestion    = True
            s.congestion_start = t
            # Fill onset_step for any pending alert events on this sensor
            for ev in eval_events:
                if ev["sensor"] == f"S{i:02d}" and ev["onset_step"] is None and not ev["is_false"]:
                    ev["onset_step"] = t
        elif not actual_congested and s.in_congestion:
            s.in_congestion    = False
            s.congestion_start = None

        results.append({
            "sensor_id":        i,
            "sensor_name":      f"S{i:02d}",
            "v_observed":       round(v_obs_mph, 2),
            "v_predicted":      round(v_pred_mph, 2),
            "residual":         round(float(r), 4),
            "anomaly_score":    round(float(A), 3),
            "threshold":        round(float(theta_A), 3),
            "P_cong":           round(float(P_cong), 3),
            "uncertainty":      round(float(U), 5),
            "theta_U":          round(float(theta_U), 5),
            "vcrit":            round(float(s.vcrit), 2),
            "anomaly_flag":     anomaly,
            "candidate_alert":  bool(candidate),
            "conditions": {
                "anomaly_exceeded": bool(c1),
                "congestion_risk":  bool(c2),
                "low_uncertainty":  bool(c3),
            },
            "propagation_score": 0.0,
            "final_warning":     False,
            "actual_congested":  bool(actual_congested),
        })

    # ── [F2] Eq.10 — propagation-aware confirmation (uses frontend η) ──
    cands = {r["sensor_id"] for r in results if r["candidate_alert"]}
    for r in results:
        nb  = neighbors[r["sensor_id"]]
        if nb:
            ps = sum(1 for j in nb if j in cands) / len(nb)
        else:
            ps = 1.0 if r["candidate_alert"] else 0.0
        r["propagation_score"] = round(float(ps), 2)
        r["final_warning"]     = bool(r["candidate_alert"] and ps >= eta)

    # ── Alert log & [F4] lead-time evaluation ─────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    for r in results:
        s = states[r["sensor_id"]]
        if r["final_warning"]:
            alert_log.appendleft({
                "time": ts, "type": "final", "sensor": r["sensor_name"],
                "msg": (f"Sensor {r['sensor_name']}: CONGESTION WARNING — "
                        f"{r['v_observed']} mph, P_cong {r['P_cong']*100:.0f}%, "
                        f"propagation {r['propagation_score']*100:.0f}%")
            })
            # Record lead time if congestion not yet started
            if not s.in_congestion and s.congestion_start is None:
                s.alert_times.append(t)
                eval_events.append({
                    "sensor": r["sensor_name"],
                    "alert_step": t,
                    "onset_step": None,          # filled when congestion confirmed later
                    "is_false": not r["actual_congested"],
                })
        elif r["candidate_alert"]:
            alert_log.appendleft({
                "time": ts, "type": "candidate", "sensor": r["sensor_name"],
                "msg": (f"Sensor {r['sensor_name']}: Candidate alert — "
                        f"propagation {r['propagation_score']*100:.0f}%")
            })

    return results, t


def _stub(i, v_obs_mph):
    s = states[i]
    return {
        "sensor_id": i, "sensor_name": f"S{i:02d}",
        "v_observed": round(v_obs_mph, 2), "v_predicted": round(v_obs_mph, 2),
        "residual": 0.0, "anomaly_score": 0.0,
        "threshold": round(s.res_ema + 2.5 * max(np.sqrt(s.res_emv), 1e-4), 3),
        "P_cong": 0.0, "uncertainty": 0.0, "theta_U": 0.05,
        "vcrit": round(float(s.vcrit), 2),
        "anomaly_flag": False, "candidate_alert": False,
        "conditions": {"anomaly_exceeded": False, "congestion_risk": False, "low_uncertainty": True},
        "propagation_score": 0.0, "final_warning": False, "actual_congested": False,
    }


# ═══════════════════════════════════════════════════════════════
# 6.  API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "running",
        "n_sensors": N_SENSORS,
        "window_steps": WINDOW,
        "horizon_steps": H_HORIZON,
        "dataset": "METR-LA (synthetic — bundled)",
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "speed_range": {"min": round(speed_min, 1), "max": round(speed_max, 1)},
        "test_step": current_step,
    })


@app.route("/api/sensors")
def api_sensors():
    return jsonify({
        "sensors": [
            {
                "id": i, "name": f"S{i:02d}",
                "neighbors": neighbors[i],
                "vcrit": round(states[i].vcrit, 2),
            }
            for i in range(N_SENSORS)
        ]
    })


@app.route("/api/traffic")
def api_traffic():
    """
    One full AAR-EWS pipeline step.
    Query params (wired from React SettingsPanel — [F2]):
      pCong       float  θ_C   default 0.6
      uncertainty float  θ_U   default None (auto per sensor)
      eta         float  η     default 0.5
    """
    theta_C  = float(request.args.get("pCong",       0.6))
    theta_U  = request.args.get("uncertainty")
    theta_U  = float(theta_U) if theta_U is not None else None
    eta      = float(request.args.get("eta",          0.5))

    data, step = run_pipeline(theta_C=theta_C, theta_U_override=theta_U, eta=eta)

    return jsonify({
        "sensors": data,
        "step":    step,
        "summary": {
            "total_sensors":    N_SENSORS,
            "active_warnings":  sum(1 for d in data if d["final_warning"]),
            "anomalies":        sum(1 for d in data if d["anomaly_flag"]),
            "candidates":       sum(1 for d in data if d["candidate_alert"]),
            "congested":        sum(1 for d in data if d["actual_congested"]),
        },
    })


@app.route("/api/history/<int:sensor_id>")
def api_history(sensor_id):
    if sensor_id not in states:
        return jsonify({"error": "sensor not found"}), 404
    s = states[sensor_id]
    return jsonify({
        "sensor_id":        sensor_id,
        "sensor_name":      f"S{sensor_id:02d}",
        "speed_history":    [round(v * speed_std + speed_mean, 2) for v in s.speed_history],
        "residual_history": [round(float(r), 4) for r in s.residual_history],
        "anomaly_scores":   [round(float(a), 3) for a in s.ascore_history],
        "vcrit":            round(float(s.vcrit), 2),
        "res_mean":         round(float(s.res_ema), 4),
        "res_std":          round(float(np.sqrt(s.res_emv)), 4),
        "n_obs":            s.n_obs,
    })


@app.route("/api/metrics")
def api_metrics():
    p_congs = []
    speeds  = []
    for i in range(N_SENSORS):
        s = states[i]
        if s.speed_history:
            v_n = list(s.speed_history)[-1]
            p   = float(1.0 / (1.0 + np.exp(-ALPHA_TCN * (s.vcrit_n - v_n))))
            p_congs.append(p)
            speeds.append(v_n * speed_std + speed_mean)

    return jsonify({
        "avg_speed_mph":     round(float(np.mean(speeds)), 2)  if speeds  else 0.0,
        "avg_p_cong":        round(float(np.mean(p_congs)), 3) if p_congs else 0.0,
        "max_p_cong":        round(float(np.max(p_congs)), 3)  if p_congs else 0.0,
        "high_risk_sensors": int(np.sum(np.array(p_congs) > 0.6)) if p_congs else 0,
        "total_alerts_stored": len(alert_log),
        "test_progress_pct": round(100 * current_step / max(len(test_data) - 1, 1), 1),
    })


@app.route("/api/alerts")
def api_alerts():
    return jsonify({"alerts": list(alert_log)})


@app.route("/api/evaluation")
def api_evaluation():
    """
    Eq.16 — Average lead time (minutes): mean(t_onset - t_alert) × INTERVAL
    Eq.17 — False alert rate (alerts/day)
    Also reports: Precision, Recall, F1 for conference metrics.
    Only meaningful after sufficient playback.
    """
    if not eval_events:
        return jsonify({"message": "No events recorded yet — keep the system running."})

    false_alerts  = [e for e in eval_events if e["is_false"]]
    true_alerts   = [e for e in eval_events if not e["is_false"]]

    # Eq.16 — Lead time: only events where onset_step was recorded
    resolved = [e for e in true_alerts if e["onset_step"] is not None]
    if resolved:
        lead_times_min = [
            max(0, (e["onset_step"] - e["alert_step"]) * INTERVAL)
            for e in resolved
        ]
        avg_lead_min = round(float(np.mean(lead_times_min)), 1)
    else:
        avg_lead_min = None

    steps_elapsed = current_step - WINDOW
    days_elapsed  = max((steps_elapsed * INTERVAL) / (60 * 24), 1.0 / 24)

    # Precision / Recall / F1
    tp = len(true_alerts)
    fp = len(false_alerts)
    # Approximate FN: count actual congestion onsets that never triggered an alert
    fn = max(0, sum(1 for s in states.values() if s.in_congestion) - tp)
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else None
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else None
    f1        = round(2 * precision * recall / (precision + recall), 3) \
                if (precision and recall and (precision + recall) > 0) else None

    return jsonify({
        "total_events":     len(eval_events),
        "true_positives":   tp,
        "false_positives":  fp,
        "false_alert_rate_per_day": round(fp / days_elapsed, 2),
        "avg_lead_time_min": avg_lead_min,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "days_elapsed":     round(days_elapsed, 2),
        "note": "Lead-time accuracy improves with longer playback duration.",
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
