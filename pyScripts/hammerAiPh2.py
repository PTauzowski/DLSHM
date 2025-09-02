# ============================================
# UCS prediction: Physics Baseline + Hybrid Residual (packed-output Keras)
# ============================================

import os, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import HuberRegressor

import keras
from keras import ops
from keras import layers, Model, regularizers
from keras.callbacks import EarlyStopping

# --------------------
# CONFIG
# --------------------
XLSX_FILE = "/Users/piotrek/Documents/Papers/Hammer/benedek-eng-diagramok.xlsx"
SHEET = "Munka1"
N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 2000
PATIENCE = 200
USE_LOG_TARGET = True  # train on log1p(UCS) and back-transform
# Physics envelopes: σc = a * ρ * W
A_LOW, A_HIGH = 0.736, 9.23

# --------------------
# Reproducibility
# --------------------
os.environ["PYTHONHASHSEED"] = "0"
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------
# Helpers
# --------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def encode_binary(series, pos_patterns):
    """Heuristic 0/1 encoder for 'saturation' / 'frozen state' columns."""
    if series is None or len(series) == 0:
        return None
    s = series.astype(str).str.lower()
    # If looks numeric, just coerce
    if s.str.match(r"^\s*[-+]?\d*\.?\d+\s*$").all():
        return pd.to_numeric(s, errors="coerce")
    out = pd.Series(0.0, index=series.index)
    mask = False
    for pat in pos_patterns:
        mask = mask | s.str.contains(pat)
    out[mask] = 1.0
    return out

def safe_to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def check_finite(name, arr):
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0][:10]
        raise ValueError(f"{name} contains non-finite values at rows {bad}")

# --------------------
# Load & clean Excel
# --------------------
if not Path(XLSX_FILE).exists():
    raise FileNotFoundError(f"Excel not found: {XLSX_FILE}")

df = pd.read_excel(XLSX_FILE, sheet_name=SHEET)

# Exact column names (adjust if needed)
COL_RHO = "density"
COL_W   = "impact work"
COL_UCS = "compressive strenght"  # (spelling per file)
COL_T   = "temperature"
COL_VP  = "Ultrasound velocity (p waves)"
COL_VS  = "ultrasound velocity of s waves "
COL_E   = "E"
COL_G   = "G"
COL_SAT = "saturation"
COL_FRO = "frozen state"

NUM_COLS = [c for c in [COL_RHO, COL_W, COL_UCS, COL_T, COL_VP, COL_VS, COL_E, COL_G] if c in df.columns]
df = safe_to_numeric(df, NUM_COLS)

# Encode saturation / frozen to 0/1
sat = encode_binary(df.get(COL_SAT, pd.Series(index=df.index, dtype=object)),
                    [r"sat", r"wet", r"nasyc", r"wod"])
froz = encode_binary(df.get(COL_FRO, pd.Series(index=df.index, dtype=object)),
                     [r"froz", r"ice", r"zamro", r"zamarz"])

df["_sat"]  = 0.0 if sat is None else sat.fillna(0.0).astype(float)
df["_froz"] = 0.0 if froz is None else froz.fillna(0.0).astype(float)

# Keep only rows with valid specimen data
required = [COL_RHO, COL_W, COL_UCS]
df = df.dropna(subset=[c for c in required if c in df.columns]).copy()

# Pull arrays (fill missing optional cols with 0)
rho = df[COL_RHO].to_numpy(float)
W   = df[COL_W].to_numpy(float)
y   = df[COL_UCS].to_numpy(float)

T  = df[COL_T].fillna(0.0).to_numpy(float)  if COL_T  in df.columns else np.zeros_like(rho)
Vp = df[COL_VP].fillna(0.0).to_numpy(float) if COL_VP in df.columns else np.zeros_like(rho)
Vs = df[COL_VS].fillna(0.0).to_numpy(float) if COL_VS in df.columns else np.zeros_like(rho)
E  = df[COL_E].fillna(0.0).to_numpy(float)  if COL_E  in df.columns else np.zeros_like(rho)
G  = df[COL_G].fillna(0.0).to_numpy(float)  if COL_G  in df.columns else np.zeros_like(rho)

sat_arr  = df["_sat"].to_numpy(float)
froz_arr = df["_froz"].to_numpy(float)

# Build features (physics-informed)
eps = 1e-6
X = np.column_stack([
    rho, W, T, Vp, Vs, E, G, sat_arr, froz_arr,
    rho * W, rho / (W + eps), (W + eps) / rho, T * sat_arr
]).astype(np.float64)

# Indices (must match X columns above)
IDX_RHO, IDX_W, IDX_SAT, IDX_FROZ = 0, 1, 7, 8

# Final sanity
check_finite("X", X)
check_finite("y", y)
print(f"Clean specimens: {len(y)} rows.")

# --------------------
# Baseline: fit k per state (no intercept) with robust loss
# --------------------
def fit_k_per_state(rho, W, y, sat, froz):
    ks = {}
    for (s, f) in [(0,0), (0,1), (1,0), (1,1)]:
        mask = (sat == s) & (froz == f)
        if mask.sum() < 3:
            ks[(s, f)] = 0.0043  # fallback (dry-normal median from paper)
            continue
        Xsf = (rho[mask] * W[mask]).reshape(-1, 1)
        reg = HuberRegressor(alpha=0.0, fit_intercept=False).fit(Xsf, y[mask])
        ks[(s, f)] = float(reg.coef_[0])
    return ks

def baseline_sigma(rho, W, sat, froz, ks):
    kvals = np.array([ks.get((int(s), int(f)), 0.0043) for s, f in zip(sat, froz)], float)
    return kvals * rho * W

# --------------------
# Model builder (MSE or Huber) — packed output [y_pred, lower, upper]
# --------------------
def build_residual_net(input_dim, lambda_bounds=1e-3, use_log_target=False, use_huber=False, huber_delta=5.0):
    """
    outputs: packed = concat([y_pred, lower_bound, upper_bound]) in the SAME domain as training target.
    loss   : data_term (MSE or Huber) + lambda_bounds * hinge_to_envelope
    """
    # Inputs
    X_in    = layers.Input(shape=(input_dim,), name="X")
    rho_in  = layers.Input(shape=(1,), name="rho")
    W_in    = layers.Input(shape=(1,), name="W")
    base_in = layers.Input(shape=(1,), name="base")  # baseline σ (MUST match training domain)

    # Tiny residual net
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-2))(X_in)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(1e-2))(x)
    delta = layers.Dense(1, activation='linear', name="delta")(x)

    # Final prediction = baseline + residual (training domain)
    y_pred = layers.Add(name="y_pred")([base_in, delta])  # (None,1)

    # Physics bounds in the SAME domain
    prod = layers.Multiply()([rho_in, W_in])  # ρW (linear)
    lower_lin = layers.Lambda(lambda t: t * A_LOW)(prod)
    upper_lin = layers.Lambda(lambda t: t * A_HIGH)(prod)
    if use_log_target:
        lower = layers.Lambda(lambda t: ops.log1p(t), name="lower")(lower_lin)
        upper = layers.Lambda(lambda t: ops.log1p(t), name="upper")(upper_lin)
    else:
        lower = layers.Identity(name="lower")(lower_lin)
        upper = layers.Identity(name="upper")(upper_lin)

    packed = layers.Concatenate(name="packed")([y_pred, lower, upper])  # (None,3)
    model = Model([X_in, rho_in, W_in, base_in], packed)

    # Custom loss
    delta_val = float(np.log1p(huber_delta)) if use_log_target else float(huber_delta)

    def hybrid_loss(y_true, packed):
        y_true = ops.reshape(y_true, (-1, 1))
        y_hat  = packed[:, 0:1]
        lower_ = packed[:, 1:2]
        upper_ = packed[:, 2:3]

        r = y_true - y_hat
        if use_huber:
            abs_r = ops.abs(r)
            data_term = ops.mean(ops.where(abs_r <= delta_val,
                                           0.5 * ops.square(r),
                                           delta_val * (abs_r - 0.5 * delta_val)))
        else:
            data_term = ops.mean(ops.square(r))

        under = ops.maximum(lower_ - y_hat, 0.0)
        over  = ops.maximum(y_hat - upper_, 0.0)
        penalty = ops.mean(under + over)

        return data_term + lambda_bounds * penalty

    model.compile(optimizer='adam', loss=hybrid_loss)
    return model

# --------------------
# Cross-validated run with compliance stats
# --------------------
def run_cv(
    X, y,
    rho_idx, w_idx, sat_idx, froz_idx,
    lambda_bounds=1e-3, use_log=True,
    use_huber=False, huber_delta=5.0,
    n_splits=N_SPLITS, random_state=RANDOM_STATE,
    use_smearing=True
):
    # Build stratification labels from X
    sat_all  = (X[:, sat_idx]  > 0.5).astype(int)
    froz_all = (X[:, froz_idx] > 0.5).astype(int)
    state = (sat_all * 2 + froz_all)  # 0..3

    skf = StratifiedKFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=random_state)

    base_MAE, base_RMSE, base_R2 = [], [], []
    hyb_MAE,  hyb_RMSE,  hyb_R2  = [], [], []

    cov_lin_list, overshoot_lin_list = [], []
    pos_meds, pos_p25, pos_p75 = [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, state), start=1):
        # 1) split
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        # 2) raw physics columns
        rho_tr, W_tr  = Xtr[:, rho_idx], Xtr[:, w_idx]
        rho_va, W_va  = Xva[:, rho_idx], Xva[:, w_idx]
        sat_tr, froz_tr = sat_all[tr], froz_all[tr]
        sat_va, froz_va = sat_all[va], froz_all[va]

        # 3) baseline (fit on TRAIN only)
        ks = fit_k_per_state(rho_tr, W_tr, ytr, sat_tr, froz_tr)
        ybase_tr = baseline_sigma(rho_tr, W_tr, sat_tr, froz_tr, ks)
        ybase_va = baseline_sigma(rho_va, W_va, sat_va, froz_va, ks)

        # baseline metrics
        base_MAE.append(mean_absolute_error(yva, ybase_va))
        base_RMSE.append(rmse(yva, ybase_va))
        base_R2.append(r2_score(yva, ybase_va))

        # 4) targets & baseline FEEDS domain
        if use_log:
            ytr_fit = np.log1p(ytr)
            yva_fit = np.log1p(yva)
            base_tr_feed = np.log1p(np.maximum(ybase_tr, 0.0))
            base_va_feed = np.log1p(np.maximum(ybase_va, 0.0))
        else:
            ytr_fit = ytr
            yva_fit = yva
            base_tr_feed = ybase_tr
            base_va_feed = ybase_va

        # 5) scale features
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr).astype(np.float32)
        Xva_s = sc.transform(Xva).astype(np.float32)

        # 6) model
        model = build_residual_net(
            input_dim=Xtr_s.shape[1],
            lambda_bounds=lambda_bounds,
            use_log_target=use_log,
            use_huber=use_huber,
            huber_delta=huber_delta
        )
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        train_feed = [
            Xtr_s,
            rho_tr[:, None].astype(np.float32),
            W_tr[:,  None].astype(np.float32),
            base_tr_feed[:, None].astype(np.float32),
        ]
        val_feed = [
            Xva_s,
            rho_va[:, None].astype(np.float32),
            W_va[:,  None].astype(np.float32),
            base_va_feed[:, None].astype(np.float32),
        ]

        history = model.fit(
            train_feed, ytr_fit.astype(np.float32),
            validation_data=(val_feed, yva_fit.astype(np.float32)),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[es]
        )

        # 7) predictions in training domain
        packed_tr = model.predict(train_feed, batch_size=BATCH_SIZE, verbose=0)
        packed_va = model.predict(val_feed,   batch_size=BATCH_SIZE, verbose=0)
        zhat_tr = packed_tr[:, 0]
        zhat_va = packed_va[:, 0]

        # Optional Duan smearing (helps log-target back-transform)
        if use_log and use_smearing:
            e = ytr_fit - zhat_tr
            S = float(np.mean(np.exp(e)))
        else:
            S = 1.0

        # Back to MPa
        yhat = np.expm1(zhat_va) if use_log else zhat_va
        if use_log and use_smearing:
            yhat = (yhat + 1.0) * S - 1.0

        # 8) physics compliance in linear domain
        lower_lin = A_LOW  * rho_va * W_va
        upper_lin = A_HIGH * rho_va * W_va
        under_lin = np.maximum(lower_lin - yhat, 0.0)
        over_lin  = np.maximum(yhat - upper_lin, 0.0)
        coverage_lin   = float(np.mean((under_lin == 0.0) & (over_lin == 0.0))) * 100.0
        overshoot_mean = float(np.mean(under_lin + over_lin))

        cov_lin_list.append(coverage_lin)
        overshoot_lin_list.append(overshoot_mean)

        # position inside envelope (0=lower, 1=upper)
        denom = (upper_lin - lower_lin)
        denom = np.where(denom == 0, np.nan, denom)
        pos = (yhat - lower_lin) / denom
        pos_meds.append(float(np.nanmedian(pos)))
        pos_p25.append(float(np.nanpercentile(pos, 25)))
        pos_p75.append(float(np.nanpercentile(pos, 75)))

        # per-state diagnostics
        print(f"[Fold {fold}]  Phys coverage={coverage_lin:.1f}% | Overshoot={overshoot_mean:.2f} MPa | "
              f"pos median={pos_meds[-1]:.2f}, IQR=[{pos_p25[-1]:.2f},{pos_p75[-1]:.2f}]")

        for (s, f), name in [((0,0), 'dry-normal'), ((0,1), 'dry-frozen'),
                             ((1,0), 'sat-normal'), ((1,1), 'sat-frozen')]:
            m = (sat_va == s) & (froz_va == f)
            if m.sum():
                mae_sf = mean_absolute_error(yva[m], yhat[m])
                rmse_sf = rmse(yva[m], yhat[m])
                print(f"   {name:12s}  MAE={mae_sf:6.2f}  RMSE={rmse_sf:6.2f}  N={m.sum():2d}")

        # hybrid metrics
        hyb_MAE.append(mean_absolute_error(yva, yhat))
        hyb_RMSE.append(rmse(yva, yhat))
        hyb_R2.append(r2_score(yva, yhat))

        print(f"[Fold {fold}]  Baseline MAE={base_MAE[-1]:.3f}  Hybrid MAE={hyb_MAE[-1]:.3f}  |  "
              f"Baseline R²={base_R2[-1]:.3f}  Hybrid R²={hyb_R2[-1]:.3f}")

    # Summary
    res = {
        "lambda": lambda_bounds,
        "log_target": bool(use_log),
        "use_huber": bool(use_huber),
        "huber_delta": huber_delta,
        "MAE_mean":  float(np.mean(hyb_MAE)),  "MAE_std":  float(np.std(hyb_MAE)),
        "RMSE_mean": float(np.mean(hyb_RMSE)), "RMSE_std": float(np.std(hyb_RMSE)),
        "R2_mean":   float(np.mean(hyb_R2)),   "R2_std":   float(np.std(hyb_R2)),
        "ΔMAE":  float(np.mean(base_MAE)  - np.mean(hyb_MAE)),
        "ΔRMSE": float(np.mean(base_RMSE) - np.mean(hyb_RMSE)),
        "ΔR2":   float(np.mean(hyb_R2)    - np.mean(base_R2)),
        "coverage_%_mean": float(np.mean(cov_lin_list)),
        "coverage_%_std":  float(np.std(cov_lin_list)),
        "overshoot_MPa_mean": float(np.mean(overshoot_lin_list)),
        "overshoot_MPa_std":  float(np.std(overshoot_lin_list)),
        "pos_median_mean": float(np.mean(pos_meds)),
        "pos_IQR25_mean":  float(np.mean(pos_p25)),
        "pos_IQR75_mean":  float(np.mean(pos_p75)),
        "baseline_MAE_mean":  float(np.mean(base_MAE)),
        "baseline_RMSE_mean": float(np.mean(base_RMSE)),
        "baseline_R2_mean":   float(np.mean(base_R2)),
    }
    print("\nPhysics compliance (linear domain): "
          f"Coverage {res['coverage_%_mean']:.1f}±{res['coverage_%_std']:.1f}% | "
          f"Overshoot {res['overshoot_MPa_mean']:.2f}±{res['overshoot_MPa_std']:.2f} MPa")
    print(f"Envelope position: median {res['pos_median_mean']:.2f}, "
          f"IQR≈[{res['pos_IQR25_mean']:.2f},{res['pos_IQR75_mean']:.2f}]")
    return res

# --------------------
# Mini grid: λ ∈ {5e-4, 1e-3, 5e-3} × {MSE, Huber}
# --------------------
lam_grid  = [5e-4, 1e-3, 5e-3]
loss_grid = [("MSE", False), ("Huber", True)]

rows = []
for lam in lam_grid:
    for loss_name, use_huber in loss_grid:
        print(f"\n=== Running CV: λ={lam}, loss={loss_name}, USE_LOG_TARGET={USE_LOG_TARGET} ===")
        res = run_cv(
            X, y,
            rho_idx=IDX_RHO, w_idx=IDX_W, sat_idx=IDX_SAT, froz_idx=IDX_FROZ,
            lambda_bounds=lam, use_log=USE_LOG_TARGET,
            use_huber=use_huber, huber_delta=5.0,
            n_splits=N_SPLITS, random_state=RANDOM_STATE,
            use_smearing=True
        )
        res["loss"] = loss_name
        rows.append(res)

grid_df = pd.DataFrame(rows)
cols = ["lambda","loss","log_target",
        "MAE_mean","MAE_std","RMSE_mean","RMSE_std","R2_mean","R2_std",
        "ΔMAE","ΔRMSE","ΔR2",
        "coverage_%_mean","overshoot_MPa_mean",
        "pos_median_mean","pos_IQR25_mean","pos_IQR75_mean"]
print("\n=== Grid summary (sorted by RMSE_mean) ===")
print(grid_df[cols].sort_values(["RMSE_mean","MAE_mean","R2_mean"], ascending=[True,True,False]).to_string(index=False))

best_idx = grid_df["RMSE_mean"].idxmin()
best = grid_df.loc[best_idx]
print("\nRecommended config:")
print(f"  lambda_bounds = {best['lambda']}, loss = {best['loss']}, log_target = {bool(best['log_target'])}")
print(f"  MAE {best['MAE_mean']:.2f}±{best['MAE_std']:.2f} | RMSE {best['RMSE_mean']:.2f}±{best['RMSE_std']:.2f} | R² {best['R2_mean']:.3f}±{best['R2_std']:.3f}")
print(f"  Coverage {best['coverage_%_mean']:.1f}% | Position median {best['pos_median_mean']:.2f} "
      f"(IQR≈[{best['pos_IQR25_mean']:.2f},{best['pos_IQR75_mean']:.2f}])")
