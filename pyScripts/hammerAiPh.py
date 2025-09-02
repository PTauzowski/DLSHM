# ============================================
# UCS prediction: Physics Baseline + Hybrid Residual (Keras)
# ============================================
# Requirements:
#   pip install pandas numpy scikit-learn tensorflow
# Optional (for pretty table): pip install tabulate

import os, random, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import HuberRegressor

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import ops

# --------------------
# CONFIG
# --------------------
XLSX_FILE = "/Users/piotrek/Documents/Papers/Hammer/benedek-eng-diagramok.xlsx"
SHEET = "Munka1"     # change if needed
N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 2000
PATIENCE = 200
USE_SOFT_BOUNDS = True       # set False to disable physics envelope penalty
USE_LOG_TARGET = True       # True = train on log1p(UCS) and back-transform
cov_lin_list, overshoot_lin_list = [], []
# Physics envelopes from paper (σc = a * ρ * W)
A_LOW, A_HIGH = 0.736, 9.23   # Benedek bounds.  (paper PDF)
#

# --------------------
# Reproducibility
# --------------------
os.environ["PYTHONHASHSEED"] = "0"
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------
# Helpers
# --------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def encode_binary(series, pos_patterns):
    """Heuristic 0/1 encoder for 'saturation' / 'frozen state' columns."""
    if series is None or len(series) == 0:
        return None
    s = series.astype(str).str.lower()
    # If it's already numeric, just coerce and return
    if s.str.match(r"^\s*[\d.]+\s*$").all():
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

# Exact column names (as in your sheet; fix here if spellings differ)
COL_RHO = "density"
COL_W   = "impact work"
COL_UCS = "compressive strenght"  # target (spelling per file)
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

# Drop non-data rows: require rho, W, UCS be numeric
required = [COL_RHO, COL_W, COL_UCS]
df = df.dropna(subset=[c for c in required if c in df.columns]).copy()

# Pull arrays
rho = df[COL_RHO].to_numpy(float)
W   = df[COL_W].to_numpy(float)
y   = df[COL_UCS].to_numpy(float)

T  = df[COL_T].fillna(0.0).to_numpy(float)  if COL_T  in df.columns else np.zeros_like(rho)
Vp = df[COL_VP].fillna(0.0).to_numpy(float) if COL_VP in df.columns else np.zeros_like(rho)
Vs = df[COL_VS].fillna(0.0).to_numpy(float) if COL_VS in df.columns else np.zeros_like(rho)
E  = df[COL_E].fillna(0.0).to_numpy(float)  if COL_E  in df.columns else np.zeros_like(rho)
G  = df[COL_G].fillna(0.0).to_numpy(float)  if COL_G  in df.columns else np.zeros_like(rho)

sat = df["_sat"].to_numpy(float)
froz= df["_froz"].to_numpy(float)

# Features (physics-informed)
eps = 1e-6
X = np.column_stack([
    rho, W, T, Vp, Vs, E, G, sat, froz,
    rho*W, rho/(W+eps), (W+eps)/rho, T*sat
]).astype(np.float64)

# Indices for convenience (must match X above)
IDX_RHO, IDX_W, IDX_SAT, IDX_FROZ = 0, 1, 7, 8

# Final sanity
check_finite("X", X)
check_finite("y", y)
print(f"Clean specimens: {len(y)} rows.")

# --------------------
# Fit k per state (no intercept) with robust loss
# --------------------
def fit_k_per_state(rho, W, y, sat, froz):
    ks = {}
    for (s,f) in [(0,0),(0,1),(1,0),(1,1)]:
        mask = (sat==s) & (froz==f)
        if mask.sum() < 3:
            ks[(s,f)] = 0.0043  # fallback (paper median for dry-normal)
            continue
        Xsf = (rho[mask] * W[mask]).reshape(-1,1)
        reg = HuberRegressor(alpha=0.0, fit_intercept=False).fit(Xsf, y[mask])
        ks[(s,f)] = float(reg.coef_[0])
    return ks

def baseline_sigma(rho, W, sat, froz, ks):
    kvals = np.array([ks.get((int(s), int(f)), 0.0043) for s,f in zip(sat, froz)], float)
    return kvals * rho * W

# --------------------
# Build residual net
# --------------------
A_LOW, A_HIGH = 0.736, 9.23  # paper envelopes

def build_residual_net(input_dim, lambda_bounds=1e-3, use_log_target=False):
    X_in    = layers.Input(shape=(input_dim,), name="X")
    rho_in  = layers.Input(shape=(1,), name="rho")
    W_in    = layers.Input(shape=(1,), name="W")
    base_in = layers.Input(shape=(1,), name="base")   # NOTE: must match the target's domain

    # Tiny residual net
    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-2))(X_in)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-2))(x)
    delta  = layers.Dense(1, activation='linear', name="delta")(x)

    # Final prediction = baseline + residual (both in SAME domain as y_true)
    y_pred = layers.Add(name="y_pred")([base_in, delta])   # (None,1)

    # Build physics bounds in the SAME domain as y_pred / y_true
    prod = layers.Multiply()([rho_in, W_in])               # ρW (linear)
    lower_lin = layers.Lambda(lambda t: t * A_LOW)(prod)
    upper_lin = layers.Lambda(lambda t: t * A_HIGH)(prod)

    if use_log_target:
        # bounds in log-domain
        lower = layers.Lambda(lambda t: ops.log1p(t), name="lower")(lower_lin)
        upper = layers.Lambda(lambda t: ops.log1p(t), name="upper")(upper_lin)
    else:
        lower = layers.Identity(name="lower")(lower_lin)
        upper = layers.Identity(name="upper")(upper_lin)

    # Pack for custom loss
    out = layers.Concatenate(name="packed")([y_pred, lower, upper])  # (None,3)

    def hybrid_loss(y_true, packed):
        y_true = ops.reshape(y_true, (-1, 1))
        y_hat  = packed[:, 0:1]
        lower_ = packed[:, 1:2]
        upper_ = packed[:, 2:3]
        # data term
        mse = ops.mean(ops.square(y_true - y_hat))
        # envelope hinge penalty
        under = ops.maximum(lower_ - y_hat, 0.0)
        over  = ops.maximum(y_hat  - upper_, 0.0)
        penalty = ops.mean(under + over)
        return mse + lambda_bounds * penalty

    model = Model([X_in, rho_in, W_in, base_in], out)
    model.compile(optimizer='adam', loss=hybrid_loss)
    return model


# --------------------
# Cross-validation (Stratified by state)
# --------------------
state = (sat.astype(int) * 2 + froz.astype(int))  # 0..3 classes
skf = StratifiedKFold(n_splits=min(N_SPLITS, len(y)), shuffle=True, random_state=RANDOM_STATE)

base_MAE, base_RMSE, base_R2 = [], [], []
hyb_MAE,  hyb_RMSE,  hyb_R2  = [], [], []

for fold, (tr, va) in enumerate(skf.split(X, state), start=1):
    # 1) Split raw data
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]

    # 2) Raw physics columns (indices must match how you built X)
    rho_tr, W_tr    = Xtr[:, IDX_RHO],  Xtr[:, IDX_W]
    rho_va, W_va    = Xva[:, IDX_RHO],  Xva[:, IDX_W]
    sat_tr, froz_tr = Xtr[:, IDX_SAT],  Xtr[:, IDX_FROZ]
    sat_va, froz_va = Xva[:, IDX_SAT],  Xva[:, IDX_FROZ]

    # 3) Fit per-state baseline k(s,f) on TRAIN ONLY, then compute baselines
    ks = fit_k_per_state(rho_tr, W_tr, ytr, sat_tr, froz_tr)
    ybase_tr = baseline_sigma(rho_tr, W_tr, sat_tr, froz_tr, ks)
    ybase_va = baseline_sigma(rho_va, W_va, sat_va, froz_va, ks)

    # 4) Targets + baseline FEEDS in the SAME domain (linear or log)
    if USE_LOG_TARGET:
        ytr_fit = np.log1p(ytr)
        yva_fit = np.log1p(yva)
        base_tr_feed = np.log1p(np.maximum(ybase_tr, 0.0))
        base_va_feed = np.log1p(np.maximum(ybase_va, 0.0))
        use_log_flag = True
    else:
        ytr_fit = ytr
        yva_fit = yva
        base_tr_feed = ybase_tr
        base_va_feed = ybase_va
        use_log_flag = False

    # 5) Scale features on TRAIN only
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Xva_s = sc.transform(Xva).astype(np.float32)

    # 6) Build & train the packed-output model
    model = build_residual_net(
        input_dim=Xtr_s.shape[1],
        lambda_bounds=1e-3,
        use_log_target=use_log_flag
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

    # --- 7a) Smearing factor from TRAIN fold (log domain) ---
    if USE_LOG_TARGET:
        # predict on TRAIN feed in log domain
        packed_tr = model.predict(train_feed, batch_size=BATCH_SIZE, verbose=0)
        zhat_tr = packed_tr[:, 0]  # log-domain predictions on train
        e = ytr_fit - zhat_tr  # log residuals
        S = float(np.mean(np.exp(e)))  # Duan smearing factor
    else:
        S = 1.0

    # --- 7b) Predict on VAL and back-transform (+ smear if log) ---
    packed_va = model.predict(val_feed, batch_size=BATCH_SIZE, verbose=0)
    zhat_va = packed_va[:, 0]  # log-domain if USE_LOG_TARGET else linear

    if USE_LOG_TARGET:
        yhat = np.expm1(zhat_va)  # to MPa
        yhat = (yhat + 1.0) * S - 1.0  # smearing correction
    else:
        yhat = zhat_va

    # Linear-domain envelope diagnostics (MPa)
    lower_lin = A_LOW * rho_va * W_va
    upper_lin = A_HIGH * rho_va * W_va

    under_lin = np.maximum(lower_lin - yhat, 0.0)
    over_lin = np.maximum(yhat - upper_lin, 0.0)

    coverage_lin = float(np.mean((under_lin == 0.0) & (over_lin == 0.0))) * 100.0  # %
    overshoot_mean = float(np.mean(under_lin + over_lin))  # MPa

    cov_lin_list.append(coverage_lin)
    overshoot_lin_list.append(overshoot_mean)

    # (Optional) per-fold print
    print(f"[Fold {fold}]  Phys coverage={coverage_lin:.1f}% | Overshoot={overshoot_mean:.2f} MPa")

    # 8) Metrics (baseline vs hybrid)
    base_MAE.append(mean_absolute_error(yva, ybase_va))
    base_RMSE.append(rmse(yva, ybase_va))
    base_R2.append(r2_score(yva, ybase_va))

    hyb_MAE.append(mean_absolute_error(yva, yhat))
    hyb_RMSE.append(rmse(yva, yhat))
    hyb_R2.append(r2_score(yva, yhat))

    if USE_LOG_TARGET and fold == 1:
        print("Examples (linear): UCS[0:5] =", ytr[:5], " baseline[0:5] =", ybase_tr[:5])
        print("Examples (log):    logUCS[0:5] =", np.log1p(ytr[:5]),
              " logBase[0:5] =", np.log1p(np.maximum(ybase_tr[:5], 0)))

    print(f"[Fold {fold}]  Baseline MAE={base_MAE[-1]:.3f}  Hybrid MAE={hyb_MAE[-1]:.3f}  |  "
          f"Baseline R²={base_R2[-1]:.3f}  Hybrid R²={hyb_R2[-1]:.3f}")


# --------------------
# Summary table
# --------------------
def mean_std(a):
    return np.mean(a), np.std(a)

b_mae, b_mae_s = mean_std(base_MAE)
b_rmse,b_rmse_s= mean_std(base_RMSE)
b_r2,  b_r2_s  = mean_std(base_R2)

h_mae, h_mae_s = mean_std(hyb_MAE)
h_rmse,h_rmse_s= mean_std(hyb_RMSE)
h_r2,  h_r2_s  = mean_std(hyb_R2)

cov_mean, cov_std = np.mean(cov_lin_list), np.std(cov_lin_list)
ovr_mean, ovr_std = np.mean(overshoot_lin_list), np.std(overshoot_lin_list)

print(f"Physics compliance (linear domain): Coverage {cov_mean:.1f}±{cov_std:.1f}% | "
      f"Overshoot {ovr_mean:.2f}±{ovr_std:.2f} MPa")

print("\n=== Cross-validated performance (mean ± std) ===")
print(f"Baseline (k per state):  MAE {b_mae:.3f}±{b_mae_s:.3f} | RMSE {b_rmse:.3f}±{b_rmse_s:.3f} | R² {b_r2:.3f}±{b_r2_s:.3f}")
print(f"Hybrid residual:         MAE {h_mae:.3f}±{h_mae_s:.3f} | RMSE {h_rmse:.3f}±{h_rmse_s:.3f} | R² {h_r2:.3f}±{h_r2_s:.3f}")
print(f"Improvement (ΔMAE):      {(b_mae - h_mae):.3f}  |  (ΔRMSE): {(b_rmse - h_rmse):.3f}  |  (ΔR²): {(h_r2 - b_r2):.3f}")
