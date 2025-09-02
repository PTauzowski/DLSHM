import numpy as np, pandas as pd, re
from keras import Input
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def k_by_state(sat, froz):
    # paper medians:
    # dry frozen=0.0037, dry normal=0.0043, saturated frozen=0.0032, saturated normal=0.0106
    if sat == 0 and froz == 1: return 0.0037
    if sat == 0 and froz == 0: return 0.0043
    if sat == 1 and froz == 1: return 0.0032
    if sat == 1 and froz == 0: return 0.0106
    return 0.0043

def baseline_sigma(rho, W, sat, froz):
    k = np.array([k_by_state(s, f) for s, f in zip(sat, froz)], dtype=float)
    return k * rho * W


TASK_PATH = '/Users/piotrek/Documents/Papers/Hammer'
XLSX_FILE = TASK_PATH + '/benedek-eng-diagramok.xlsx'

SHEET = "Munka1"
df = pd.read_excel(XLSX_FILE, sheet_name=SHEET)

# ---- exact column names from your snippet (fix typos to match Excel) ----
COL_RHO = 'density'
COL_W   = 'impact work'
COL_UCS = 'compressive strenght'  # target
COL_T   = 'temperature'
COL_VP  = 'Ultrasound velocity (p waves)'
COL_VS  = 'ultrasound velocity of s waves '
COL_E   = 'E'
COL_G   = 'G'
COL_SAT = 'saturation'
COL_FRO = 'frozen state'

# 1) coerce ALL numeric columns you will use (including TARGET)
num_cols = [c for c in [COL_RHO, COL_W, COL_UCS, COL_T, COL_VP, COL_VS, COL_E, COL_G] if c in df.columns]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 2) encode categorical flags (0/1). If the column is numeric already, keep it.
def bin_encode(series, pos_patterns):
    if series.name not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=float))
    s = series.astype(str).str.lower()
    out = pd.Series(0.0, index=series.index)
    mask = False
    for pat in pos_patterns:
        mask = mask | s.str.contains(pat)
    out[mask] = 1.0
    return out

# Tiny residual net
def build_residual_net(input_dim):
    m = Sequential([
        Dense(16, activation='relu', kernel_regularizer=l2(1e-2), input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(8, activation='relu',  kernel_regularizer=l2(1e-2)),
        Dense(1, activation='linear')
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

sat = bin_encode(df.get(COL_SAT, pd.Series(index=df.index, dtype=object)),
                 pos_patterns=[r'sat', r'wet', r'nasyc', r'wod'])
froz = bin_encode(df.get(COL_FRO, pd.Series(index=df.index, dtype=object)),
                  pos_patterns=[r'froz', r'ice', r'zamro', r'zamarz'])

df['_sat']  = sat
df['_froz'] = froz

# 3) drop any row that is not a specimen (i.e., any NaN in REQUIRED cols)
required = [COL_RHO, COL_W, COL_UCS]  # minimally needed
df = df.dropna(subset=[c for c in required if c in df.columns])

# 4) now safely build features (guard divides)
eps = 1e-6
rho = df[COL_RHO].to_numpy(float)
W   = df[COL_W].to_numpy(float)
UCS = df[COL_UCS].to_numpy(float)

T  = df[COL_T].fillna(0.0).to_numpy(float)  if COL_T  in df.columns else np.zeros_like(rho)
Vp = df[COL_VP].fillna(0.0).to_numpy(float) if COL_VP in df.columns else np.zeros_like(rho)
Vs = df[COL_VS].fillna(0.0).to_numpy(float) if COL_VS in df.columns else np.zeros_like(rho)
E  = df[COL_E].fillna(0.0).to_numpy(float)  if COL_E  in df.columns else np.zeros_like(rho)
G  = df[COL_G].fillna(0.0).to_numpy(float)  if COL_G  in df.columns else np.zeros_like(rho)

sat = df['_sat'].to_numpy(float)
froz= df['_froz'].to_numpy(float)

X = np.column_stack([
    rho, W, T, Vp, Vs, E, G, sat, froz,
    rho*W, rho/(W+eps), (W+eps)/rho, T*sat
]).astype(np.float64)

y = UCS.astype(np.float64)

# 5) final sanity checks (catch any NaN/Inf before CV)
def _ok(arr, name):
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0][:10]
        raise ValueError(f"{name} contains non-finite values at rows {bad[:10]}")
_ok(X, "X")
_ok(y, "y")

print(f"Clean specimens: {len(y)} rows. No NaN/Inf in X or y.")

# --- small MLP builder ---
def build():
    m = Sequential([
        # Dense(16, activation='relu', kernel_regularizer=l2(1e-2), input_shape=(X.shape[1],)),
        # Dropout(0.2),
        # Dense(8, activation='relu', kernel_regularizer=l2(1e-2)),
        # Dense(1, activation='linear')
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(1e-3)),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

# --- K-fold CV ---
kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
maes, rmses, r2s = [], [], []

for tr, va in kf.split(X):
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]

    # Get columns for baseline (rho, W, sat, froz)
    rho_tr, W_tr = Xtr[:, 0], Xtr[:, 1]
    rho_va, W_va = Xva[:, 0], Xva[:, 1]
    sat_tr, froz_tr = Xtr[:, 7], Xtr[:, 8]
    sat_va, froz_va = Xva[:, 7], Xva[:, 8]

    # Baseline sigma for this fold
    ybase_tr = baseline_sigma(rho_tr, W_tr, sat_tr, froz_tr)
    ybase_va = baseline_sigma(rho_va, W_va, sat_va, froz_va)

    # Residual target
    ytr_res = ytr - ybase_tr

    # Scale features (fit on train only)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

    # Train residual net on residuals
    resnet = build_residual_net(X.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True)
    resnet.fit(Xtr_s, ytr_res,
               validation_data=(Xva_s, yva - ybase_va),
               epochs=2000,
               batch_size=16,
               verbose=1,
               callbacks=[es])

    # Predict UCS
    yhat_va = ybase_va + resnet.predict(Xva_s, batch_size=16).ravel()
    maes.append(mean_absolute_error(yva, yhat_va))
    rmses.append(np.sqrt(((yva - yhat_va)**2).mean()))
    r2s.append(r2_score(yva, yhat_va))

print(f"Small MLP — MAE: {np.mean(maes):.3f}, RMSE: {np.mean(rmses):.3f}, R^2: {np.mean(r2s):.3f}")
