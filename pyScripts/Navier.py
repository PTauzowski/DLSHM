import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 0. General settings
# ============================================================

ν = 0.01  # kinematic viscosity

nx = ny = 30                # interior grid in each direction
h = 1.0 / (nx + 1)
N_int = nx * ny

# ============================================================
# 1. Interior and boundary points
# ============================================================

# Interior collocation points as a regular grid
xs = np.linspace(h, 1.0 - h, nx)
ys = np.linspace(h, 1.0 - h, ny)
X_int, Y_int = np.meshgrid(xs, ys, indexing="ij")
X_int_flat = X_int.reshape(-1, 1)
Y_int_flat = Y_int.reshape(-1, 1)
XY_int = np.hstack([X_int_flat, Y_int_flat])          # (N_int, 2)

# For this example, body force f = 0
F1_int = np.zeros((N_int, 1), dtype=np.float32)
F2_int = np.zeros((N_int, 1), dtype=np.float32)

# Boundary points: random sampling on all four edges (Dirichlet u=v=0)
nb = 1000
xy_rand = np.random.rand(nb, 2)
edge = np.random.randint(0, 4, size=(nb, 1))

Xb = xy_rand.copy()
# left: x=0
Xb[edge[:, 0] == 0, 0] = 0.0
# right: x=1
Xb[edge[:, 0] == 1, 0] = 1.0
# bottom: y=0
Xb[edge[:, 0] == 2, 1] = 0.0
# top: y=1
Xb[edge[:, 0] == 3, 1] = 1.0

Ub = np.zeros((nb, 2), dtype=np.float32)   # u=v=0 at boundary

# Convert to tf tensors
XY_int_tf = tf.constant(XY_int, dtype=tf.float32)     # (N_int,2)
F1_int_tf = tf.constant(F1_int, dtype=tf.float32)     # (N_int,1)
F2_int_tf = tf.constant(F2_int, dtype=tf.float32)     # (N_int,1)

Xb_tf = tf.constant(Xb, dtype=tf.float32)             # (nb,2)
Ub_tf = tf.constant(Ub, dtype=tf.float32)             # (nb,2)


# ============================================================
# 2. Gram matrix G and its inverse (scalar 5-point Laplacian)
#    We reuse it for each scalar residual (u-momentum, v-momentum, div)
# ============================================================

def build_gram_matrix_dense(nx, ny, h):
    N = nx * ny
    G = np.zeros((N, N), dtype=np.float64)

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            G[k, k] += 4.0

            if i > 0:
                G[k, idx(i - 1, j)] -= 1.0
            if i < nx - 1:
                G[k, idx(i + 1, j)] -= 1.0
            if j > 0:
                G[k, idx(i, j - 1)] -= 1.0
            if j < ny - 1:
                G[k, idx(i, j + 1)] -= 1.0

    G *= 1.0 / (h * h)
    return G

G_np = build_gram_matrix_dense(nx, ny, h)          # (N_int, N_int)
G_inv_np = np.linalg.inv(G_np)                     # OK for moderate grids
G_inv_tf = tf.constant(G_inv_np.astype(np.float32))


def gram_loss(res_flat):
    """
    res_flat: (N_int,) tensor
    returns res^T G^{-1} res
    """
    tmp = tf.linalg.matvec(G_inv_tf, res_flat)     # (N_int,)
    return tf.tensordot(res_flat, tmp, axes=1)     # scalar


# ============================================================
# 3. PINN model: (x, y) -> (u, v, p)
# ============================================================

def build_ns_pinn(hidden_layers=5, units=64):
    inp = keras.Input(shape=(2,))
    x = inp
    for _ in range(hidden_layers):
        x = layers.Dense(units, activation="tanh")(x)
    out = layers.Dense(3, activation=None)(x)      # [u, v, p]
    return keras.Model(inputs=inp, outputs=out)

model = build_ns_pinn()


# ============================================================
# 4. Compute residuals via automatic differentiation
# ============================================================

@tf.function
def ns_residuals(model, XY, F1, F2):
    """
    XY : (N_int,2) interior points
    F1,F2 : (N_int,1) body forces (here zeros)
    Returns r_u, r_v, r_div (each (N_int,1))
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(XY)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(XY)
            uvp = model(XY)                 # (N_int,3)
            u = uvp[:, 0:1]
            v = uvp[:, 1:2]
            p = uvp[:, 2:3]

        # first derivatives
        grads = tape1.gradient(uvp, XY)     # (N_int,2) each component's d/dx + d/dy summed, so we re-compute separately
        # better: compute per component:
        du_dXY = tape1.gradient(u, XY)      # (N_int,2)
        dv_dXY = tape1.gradient(v, XY)      # (N_int,2)
        dp_dXY = tape1.gradient(p, XY)      # (N_int,2)

        du_dx = du_dXY[:, 0:1]
        du_dy = du_dXY[:, 1:2]
        dv_dx = dv_dXY[:, 0:1]
        dv_dy = dv_dXY[:, 1:2]

        dp_dx = dp_dXY[:, 0:1]
        dp_dy = dp_dXY[:, 1:2]

    # second derivatives for Laplacian
    d2u_dx2 = tape2.gradient(du_dx, XY)[:, 0:1]
    d2u_dy2 = tape2.gradient(du_dy, XY)[:, 1:2]
    d2v_dx2 = tape2.gradient(dv_dx, XY)[:, 0:1]
    d2v_dy2 = tape2.gradient(dv_dy, XY)[:, 1:2]
    del tape1
    del tape2

    lap_u = d2u_dx2 + d2u_dy2
    lap_v = d2v_dx2 + d2v_dy2

    # convective term u·∇u
    conv_u = u * du_dx + v * du_dy
    conv_v = u * dv_dx + v * dv_dy

    # momentum residuals: u·∇u - νΔu + ∇p - f = 0
    r_u = conv_u - ν * lap_u + dp_dx - F1
    r_v = conv_v - ν * lap_v + dp_dy - F2

    # incompressibility residual: div u = 0
    r_div = du_dx + dv_dy

    return r_u, r_v, r_div


# ============================================================
# 5. Total loss = CRVPINN PDE loss + BC loss
# ============================================================

@tf.function
def compute_total_loss(model, XY_int, F1_int, F2_int, Xb, Ub,
                       beta_div=1.0, w_bc=1.0):
    r_u, r_v, r_div = ns_residuals(model, XY_int, F1_int, F2_int)

    r_u_flat = tf.reshape(r_u, (-1,))
    r_v_flat = tf.reshape(r_v, (-1,))
    r_div_flat = tf.reshape(r_div, (-1,))

    loss_pde = gram_loss(r_u_flat) + gram_loss(r_v_flat) \
               + beta_div * gram_loss(r_div_flat)

    # boundary loss: u,v ≈ 0
    uvb = model(Xb)[:, 0:2]                # (nb,2)
    loss_bc = tf.reduce_mean(tf.square(uvb - Ub))

    total_loss = loss_pde + w_bc * loss_bc
    return total_loss, loss_pde, loss_bc


# ============================================================
# 6. Training loop
# ============================================================

optimizer = keras.optimizers.Adam(1e-3)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        total_loss, loss_pde, loss_bc = compute_total_loss(
            model,
            XY_int_tf, F1_int_tf, F2_int_tf,
            Xb_tf, Ub_tf,
            beta_div=10.0,   # strong divergence penalty
            w_bc=10.0        # strong boundary enforcement
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_pde, loss_bc

epochs = 5000
for epoch in range(1, epochs + 1):
    total_loss, loss_pde, loss_bc = train_step()
    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Total = {total_loss.numpy():.4e}, "
            f"PDE = {loss_pde.numpy():.4e}, "
            f"BC = {loss_bc.numpy():.4e}"
        )

# After training you can sample the domain and plot u,v,p similarly to the slide.
