"""Matrix sign function implementations in JAX.

Ports of:
- Newton-Schulz 5 (Keller Jordan, github.com/KellerJordan/Muon)
- Polar Express (Amsel, Persson, Musco, Gower, arXiv:2505.16932;
    github.com/thinking-machines-lab/manifolds)
"""

import jax
import jax.numpy as jnp

# Muon NS5: fixed quintic coefficients maximizing slope at zero.
_NS5_ABC = (3.4445, -4.7750, 2.0315)

# Polar Express: step-dependent coefficients for faster convergence.
_POLAR_EXPRESS_ABC = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# Safety factor for numerical stability (exclude last polynomial).
_POLAR_EXPRESS_ABC_STABLE = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in _POLAR_EXPRESS_ABC[:-1]
] + [_POLAR_EXPRESS_ABC[-1]]


def _extract_trace(X, L, R):
  """Extract diagonal and off-diagonal energy of L^T X R."""
  M = L.mT @ X.astype(jnp.float32) @ R
  diag = jnp.diag(M)
  offdiag = jnp.linalg.norm(M - jnp.diag(diag))
  return diag, offdiag


def _setup(G, U, V):
  """Shared setup: transpose if tall, select SVD bases accordingly."""
  X = G.astype(jnp.bfloat16)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT
    L, R = V, U
  else:
    L, R = U, V
  return X, L, R, transposed


@jax.jit(static_argnames=("steps",))
def newtonschulz5(G, steps=5):
  """Newton-Schulz iteration with fixed quintic coefficients (Muon-style).

  Computes an approximate polar factor of G in bfloat16.
  Returns bfloat16.
  """
  assert G.ndim >= 2
  a, b, c = _NS5_ABC
  X = G.astype(jnp.bfloat16)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT

  X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + 1e-7)

  for _ in range(steps):
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X

  if transposed:
    X = X.mT
  return X


@jax.jit(static_argnames=("steps",))
def newtonschulz5_traced(G, U, V, steps=5):
  """NS5 with per-step singular value and subspace mixing tracking."""
  assert G.ndim == 2
  a, b, c = _NS5_ABC
  X, L, R, transposed = _setup(G, U, V)
  min_dim = min(G.shape)

  X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + 1e-7)

  sigmas = jnp.zeros((steps + 1, min_dim))
  offdiags = jnp.zeros(steps + 1)
  d, od = _extract_trace(X, L, R)
  sigmas = sigmas.at[0].set(d)
  offdiags = offdiags.at[0].set(od)

  for i in range(steps):
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X
    d, od = _extract_trace(X, L, R)
    sigmas = sigmas.at[i + 1].set(d)
    offdiags = offdiags.at[i + 1].set(od)

  if transposed:
    X = X.mT
  return X, sigmas, offdiags


@jax.jit(static_argnames=("steps",))
def polar_express(G, steps=10):
  """Polar Express matrix sign function.

  Uses step-dependent polynomial coefficients for faster convergence.
  Returns bfloat16.
  """
  assert G.ndim >= 2
  X = G.astype(jnp.bfloat16)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT

  X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) * 1.01)
  I = jnp.eye(X.shape[-2], dtype=X.dtype)

  for step in range(steps):
    idx = min(step, len(_POLAR_EXPRESS_ABC_STABLE) - 1)
    a, b, c = _POLAR_EXPRESS_ABC_STABLE[idx]
    S = X @ X.mT
    Y = c * S + b * I
    Y = Y @ S + a * I
    X = Y @ X

  if transposed:
    X = X.mT
  return X


@jax.jit(static_argnames=("steps",))
def polar_express_traced(G, U, V, steps=10):
  """Polar Express with per-step singular value and subspace mixing tracking."""
  assert G.ndim == 2
  X, L, R, transposed = _setup(G, U, V)
  min_dim = min(G.shape)

  X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) * 1.01)
  I = jnp.eye(X.shape[-2], dtype=X.dtype)

  sigmas = jnp.zeros((steps + 1, min_dim))
  offdiags = jnp.zeros(steps + 1)
  d, od = _extract_trace(X, L, R)
  sigmas = sigmas.at[0].set(d)
  offdiags = offdiags.at[0].set(od)

  for step in range(steps):
    idx = min(step, len(_POLAR_EXPRESS_ABC_STABLE) - 1)
    a, b, c = _POLAR_EXPRESS_ABC_STABLE[idx]
    S = X @ X.mT
    Y = c * S + b * I
    Y = Y @ S + a * I
    X = Y @ X
    d, od = _extract_trace(X, L, R)
    sigmas = sigmas.at[step + 1].set(d)
    offdiags = offdiags.at[step + 1].set(od)

  if transposed:
    X = X.mT
  return X, sigmas, offdiags
