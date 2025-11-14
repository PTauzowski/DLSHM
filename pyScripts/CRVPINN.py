import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. Problem setup: Poisson on (0,1)^2 with known RHS
# ============================================================

# Example exact solution and RHS: u(x,y) = sin(pi x) sin(pi y)
# Then -Δu = 2 pi^2 sin(pi x) sin(pi y)
def exact_u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def rhs_f(x, y):
    return 2.0 * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)


# ============================================================
# 2. Generate collocation and boundary points
# ============================================================

nx = ny = 30  # interior grid in each direction
h = 1.0 / (nx + 1)  # spacing; interior points exclude boundary
N_int = nx * ny  # number of interior points

# Interior points (x_i, y_j) with i,j = 1..nx,ny
xs = np.linspace(h, 1.0 - h, nx)
ys = np.linspace(h, 1.0 - h, ny)
X_int, Y_int = np.meshgrid(xs, ys, indexing="ij")
X_int_flat = X_int.reshape(-1, 1)
Y_int_flat = Y_int.reshape(-1, 1)

# RHS on interior
F_int_flat = rhs_f(X_int_flat, Y_int_flat)

# Boundary points for Dirichlet BCs
nb = 1000  # random boundary samples
xb = np.random.rand(nb, 1)
yb = np.random.rand(nb, 1)
# pick which edge: 0=left,1=right,2=bottom,3=top
edge = np.random.randint(0, 4, size=(nb, 1))

xb_bc = xb.copy()
yb_bc = yb.copy()
# left x=0
xb_bc[edge == 0] = 0.0
# right x=1
xb_bc[edge == 1] = 1.0
# bottom y=0
yb_bc[edge == 2] = 0.0
# top y=1
yb_bc[edge == 3] = 1.0

U_bc = exact_u(xb_bc, yb_bc)  # Dirichlet values on boundary

# Convert all to float32 tensors
X_int_tf = tf.constant(np.hstack([X_int_flat, Y_int_flat]), dtype=tf.float32)  # shape (N_int, 2)
F_int_tf = tf.constant(F_int_flat, dtype=tf.float32)  # shape (N_int, 1)

Xb_tf = tf.constant(np.hstack([xb_bc, yb_bc]), dtype=tf.float32)  # shape (nb, 2)
Ub_tf = tf.constant(U_bc, dtype=tf.float32)  # shape (nb, 1)


# ============================================================
# 3. Build Gram matrix G and its inverse G_inv
#    (5-point Laplacian on Nx-by-Ny grid)
# ============================================================

def build_gram_matrix_dense(nx, ny, h):
    N = nx * ny
    G = np.zeros((N, N), dtype=np.float64)

    def idx(i, j):
        return i * ny + j  # 0-based index

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

    G *= (1.0 / (h * h))
    return G


G_np = build_gram_matrix_dense(nx, ny, h)  # (N_int, N_int)
# Invert once (OK for moderate N)
G_inv_np = np.linalg.inv(G_np)  # (N_int, N_int)

# Store as tf constant for use in loss
G_inv_tf = tf.constant(G_inv_np.astype(np.float32))  # (N_int, N_int)


# ============================================================
# 4. Build Keras model u_theta(x,y)
# ============================================================

def build_pinn(hidden_layers=4, units=64):
    inputs = keras.Input(shape=(2,))
    x = inputs
    for _ in range(hidden_layers):
        x = layers.Dense(units, activation="tanh")(x)
    outputs = layers.Dense(1, activation=None)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_pinn(hidden_layers=4, units=64)


# ============================================================
# 5. Helper: compute Laplacian of u_theta via auto-diff
# ============================================================

@tf.function
def laplacian_u(model, x):
    """
    x: (N,2) tensor with columns [x, y].
    returns: (N,1) tensor of Δu(x,y).
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            u = model(x)  # (N,1)
        # first derivatives
        du_dx_dy = tape1.gradient(u, x)  # (N,2)
        du_dx = du_dx_dy[:, 0:1]
        du_dy = du_dx_dy[:, 1:2]
    # second derivatives
    d2u_dx2 = tape2.gradient(du_dx, x)[:, 0:1]  # (N,1)
    d2u_dy2 = tape2.gradient(du_dy, x)[:, 1:2]  # (N,1)
    del tape1
    del tape2
    lap = d2u_dx2 + d2u_dy2
    return lap


# ============================================================
# 6. CRVPINN loss computation
# ============================================================

@tf.function
def compute_losses(model, X_int, F_int, Xb, Ub):
    """
    X_int: (N_int,2), interior points
    F_int: (N_int,1), RHS f(x,y)
    Xb:    (Nb,2), boundary points
    Ub:    (Nb,1), boundary values
    """
    # PDE residual at interior: -Δu - f = 0  => residual = (-Δu - f)
    lap = laplacian_u(model, X_int)  # (N_int,1)
    res = -lap - F_int  # (N_int,1)
    res_flat = tf.reshape(res, (-1,))  # (N_int,)

    # CRVPINN robust loss: res^T G_inv res
    # note: G_inv_tf is (N_int, N_int) constant
    tmp = tf.linalg.matvec(G_inv_tf, res_flat)  # (N_int,)
    loss_pde = tf.tensordot(res_flat, tmp, axes=1)  # scalar

    # Boundary loss: simple MSE
    u_b = model(Xb)  # (Nb,1)
    loss_bc = tf.reduce_mean(tf.square(u_b - Ub))

    # Total loss (weights can be tuned)
    total_loss = loss_pde + 1.0 * loss_bc
    return total_loss, loss_pde, loss_bc


# ============================================================
# 7. Optimizer and custom training loop
# ============================================================

optimizer = keras.optimizers.Adam(learning_rate=1e-3)


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        total_loss, loss_pde, loss_bc = compute_losses(
            model, X_int_tf, F_int_tf, Xb_tf, Ub_tf
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_pde, loss_bc


# Training loop
epochs = 5000
for epoch in range(1, epochs + 1):
    total_loss, loss_pde, loss_bc = train_step()
    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Loss total = {total_loss.numpy():.4e}, "
            f"PDE = {loss_pde.numpy():.4e}, "
            f"BC = {loss_bc.numpy():.4e}"
        )

# ============================================================
# 8. Optional: check error vs exact solution on a grid
# ============================================================

# validation grid
nxv = nyv = 40
xs_v = np.linspace(0, 1, nxv)
ys_v = np.linspace(0, 1, nyv)
Xv, Yv = np.meshgrid(xs_v, ys_v, indexing="ij")
xy_v = np.hstack([Xv.reshape(-1, 1), Yv.reshape(-1, 1)])

u_pred = model(tf.constant(xy_v, dtype=tf.float32)).numpy().reshape(nxv, nyv)
u_true = exact_u(Xv, Yv)
l2_err = np.sqrt(np.mean((u_pred - u_true) ** 2))
print("L2 error vs exact solution:", l2_err)
