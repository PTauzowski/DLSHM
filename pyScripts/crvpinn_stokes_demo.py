import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ============================================================
# 0. Settings
# ============================================================

nu = 0.01          # viscosity
nx = ny = 20       # interior grid size (small -> fast demo)
epochs = 5000      # training iterations

print("TensorFlow:", tf.__version__)


# ============================================================
# 1. Manufactured Stokes solution on (0,1)^2
# ============================================================

def manufactured_solution(XY):
    """
    Manufactured (u,v,p) that satisfies incompressibility and Stokes.
    XY: (N,2) tensor [x,y] with entries in [0,1].
    Stream function psi = sin(pi x) sin(pi y):
        u = dpsi/dy =  pi cos(pi y) sin(pi x)
        v = -dpsi/dx = -pi cos(pi x) sin(pi y)
        p = sin(pi x) sin(pi y)
    """
    x = XY[:, 0:1]
    y = XY[:, 1:2]
    pi = tf.constant(np.pi, dtype=XY.dtype)

    u = pi * tf.cos(pi * y) * tf.sin(pi * x)
    v = -pi * tf.cos(pi * x) * tf.sin(pi * y)
    p = tf.sin(pi * x) * tf.sin(pi * y)
    return u, v, p


# ============================================================
# 2. Interior points and body force f
# ============================================================

h = 1.0 / (nx + 1)
xs = np.linspace(h, 1.0 - h, nx)
ys = np.linspace(h, 1.0 - h, ny)
X_int, Y_int = np.meshgrid(xs, ys, indexing="ij")
XY_int = np.hstack([X_int.reshape(-1, 1),
                    Y_int.reshape(-1, 1)]).astype(np.float32)
N_int = XY_int.shape[0]

XY_int_tf = tf.constant(XY_int, dtype=tf.float32)
XY_int_var = tf.Variable(XY_int_tf)

# Compute f = -nu Δu_exact + ∇p_exact using autodiff
with tf.GradientTape(persistent=True) as tape2:
    tape2.watch(XY_int_var)
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(XY_int_var)
        u_ex, v_ex, p_ex = manufactured_solution(XY_int_var)

    du_dXY = tape1.gradient(u_ex, XY_int_var)
    dv_dXY = tape1.gradient(v_ex, XY_int_var)
    dp_dXY = tape1.gradient(p_ex, XY_int_var)

    du_dx = du_dXY[:, 0:1]
    du_dy = du_dXY[:, 1:2]
    dv_dx = dv_dXY[:, 0:1]
    dv_dy = dv_dXY[:, 1:2]
    dp_dx = dp_dXY[:, 0:1]
    dp_dy = dp_dXY[:, 1:2]

d2u_dx2 = tape2.gradient(du_dx, XY_int_var)[:, 0:1]
d2u_dy2 = tape2.gradient(du_dy, XY_int_var)[:, 1:2]
d2v_dx2 = tape2.gradient(dv_dx, XY_int_var)[:, 0:1]
d2v_dy2 = tape2.gradient(dv_dy, XY_int_var)[:, 1:2]

del tape1, tape2

lap_u_ex = d2u_dx2 + d2u_dy2
lap_v_ex = d2v_dx2 + d2v_dy2

f1 = -nu * lap_u_ex + dp_dx
f2 = -nu * lap_v_ex + dp_dy

F1_int = f1.numpy().astype(np.float32)
F2_int = f2.numpy().astype(np.float32)
U_int_exact = u_ex.numpy().astype(np.float32)
V_int_exact = v_ex.numpy().astype(np.float32)
P_int_exact = p_ex.numpy().astype(np.float32)

XY_int_tf = tf.constant(XY_int, dtype=tf.float32)
F1_int_tf = tf.constant(F1_int, dtype=tf.float32)
F2_int_tf = tf.constant(F2_int, dtype=tf.float32)


# ============================================================
# 3. Boundary points and exact Dirichlet data
# ============================================================

nb = 400
xy_rand = np.random.rand(nb, 2).astype(np.float32)
edge = np.random.randint(0, 4, size=(nb, 1))

Xb = xy_rand.copy()
Xb[edge[:, 0] == 0, 0] = 0.0  # left
Xb[edge[:, 0] == 1, 0] = 1.0  # right
Xb[edge[:, 0] == 2, 1] = 0.0  # bottom
Xb[edge[:, 0] == 3, 1] = 1.0  # top

Xb_tf = tf.constant(Xb, dtype=tf.float32)
u_b_ex, v_b_ex, p_b_ex = manufactured_solution(Xb_tf)
Ub_tf = tf.concat([u_b_ex, v_b_ex], axis=1)  # (nb,2)


# ============================================================
# 4. Gram matrix and its inverse (for CRVPINN loss)
# ============================================================

def build_gram_matrix_dense(nx, ny, h):
    """
    5-point Laplacian stiffness matrix on regular grid.
    """
    N = nx * ny
    G = np.zeros((N, N), dtype=np.float64)

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            G[k, k] += 4.0
            if i > 0:       G[k, idx(i - 1, j)] -= 1.0
            if i < nx - 1:  G[k, idx(i + 1, j)] -= 1.0
            if j > 0:       G[k, idx(i, j - 1)] -= 1.0
            if j < ny - 1:  G[k, idx(i, j + 1)] -= 1.0

    G *= 1.0 / (h * h)
    return G

G_np = build_gram_matrix_dense(nx, ny, h)
G_inv_np = np.linalg.inv(G_np)                     # OK for small demo
G_inv_tf = tf.constant(G_inv_np.astype(np.float32))

def gram_loss(res_flat):
    """
    res_flat: (N,) tensor -> res^T G^{-1} res
    """
    tmp = tf.linalg.matvec(G_inv_tf, res_flat)
    return tf.tensordot(res_flat, tmp, axes=1)


# ============================================================
# 5. PINN model (x,y) -> (u,v,p)
# ============================================================

def build_stokes_pinn(hidden_layers=4, units=32):
    inp = keras.Input(shape=(2,))
    x = inp
    for _ in range(hidden_layers):
        x = layers.Dense(units, activation="tanh")(x)
    out = layers.Dense(3, activation=None)(x)
    return keras.Model(inputs=inp, outputs=out)

model = build_stokes_pinn()
optimizer = keras.optimizers.Adam(1e-3)


# ============================================================
# 6. Residuals and loss functions
# ============================================================

@tf.function
def stokes_residuals(model, XY, F1, F2):
    """
    Returns r_u, r_v, r_div at interior points.
    Stokes: -nu Δu + ∇p = f,  div u = 0
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(XY)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(XY)
            uvp = model(XY)
            u = uvp[:, 0:1]
            v = uvp[:, 1:2]
            p = uvp[:, 2:3]

        du_dXY = tape1.gradient(u, XY)
        dv_dXY = tape1.gradient(v, XY)
        dp_dXY = tape1.gradient(p, XY)

        du_dx = du_dXY[:, 0:1]
        du_dy = du_dXY[:, 1:2]
        dv_dx = dv_dXY[:, 0:1]
        dv_dy = dv_dXY[:, 1:2]
        dp_dx = dp_dXY[:, 0:1]
        dp_dy = dp_dXY[:, 1:2]

    d2u_dx2 = tape2.gradient(du_dx, XY)[:, 0:1]
    d2u_dy2 = tape2.gradient(du_dy, XY)[:, 1:2]
    d2v_dx2 = tape2.gradient(dv_dx, XY)[:, 0:1]
    d2v_dy2 = tape2.gradient(dv_dy, XY)[:, 1:2]

    del tape1, tape2

    lap_u = d2u_dx2 + d2u_dy2
    lap_v = d2v_dx2 + d2v_dy2

    r_u = -nu * lap_u + dp_dx - F1
    r_v = -nu * lap_v + dp_dy - F2
    r_div = du_dx + dv_dy

    return r_u, r_v, r_div


@tf.function
def compute_total_loss(model, XY_int, F1_int, F2_int, Xb, Ub,
                       beta_div=10.0, w_bc=10.0):
    r_u, r_v, r_div = stokes_residuals(model, XY_int, F1_int, F2_int)
    r_u_flat = tf.reshape(r_u, (-1,))
    r_v_flat = tf.reshape(r_v, (-1,))
    r_div_flat = tf.reshape(r_div, (-1,))

    loss_pde = gram_loss(r_u_flat) + gram_loss(r_v_flat) \
               + beta_div * gram_loss(r_div_flat)

    uvb = model(Xb)[:, 0:2]
    loss_bc = tf.reduce_mean(tf.square(uvb - Ub))

    total = loss_pde + w_bc * loss_bc
    return total, loss_pde, loss_bc


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        total_loss, loss_pde, loss_bc = compute_total_loss(
            model, XY_int_tf, F1_int_tf, F2_int_tf, Xb_tf, Ub_tf
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_pde, loss_bc


# ============================================================
# 7. Training loop
# ============================================================

for epoch in range(1, epochs + 1):
    total_loss, loss_pde, loss_bc = train_step()
    if epoch % 300 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | "
              f"total={total_loss.numpy():.3e}, "
              f"PDE={loss_pde.numpy():.3e}, "
              f"BC={loss_bc.numpy():.3e}")


# ============================================================
# 8. Evaluate on grid and plot PINN vs exact
# ============================================================

XY_plot = XY_int
XY_plot_tf = tf.constant(XY_plot, dtype=tf.float32)
uvp_pred = model(XY_plot_tf).numpy()

u_pred = uvp_pred[:, 0].reshape(nx, ny)
v_pred = uvp_pred[:, 1].reshape(nx, ny)
p_pred = uvp_pred[:, 2].reshape(nx, ny)

u_true = U_int_exact.reshape(nx, ny)
v_true = V_int_exact.reshape(nx, ny)
p_true = P_int_exact.reshape(nx, ny)

u_err = np.abs(u_pred - u_true)
v_err = np.abs(v_pred - v_true)
p_err = np.abs(p_pred - p_true)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

titles = [
    "u (PINN)", "u (exact)", "u abs error",
    "v (PINN)", "v (exact)", "v abs error",
    "p (PINN)", "p (exact)", "p abs error",
]
fields = [
    u_pred, u_true, u_err,
    v_pred, v_true, v_err,
    p_pred, p_true, p_err,
]

for ax, field, title in zip(axes.flatten(), fields, titles):
    im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
