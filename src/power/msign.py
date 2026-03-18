"""Matrix sign function implementations in JAX.

Ports of:
- Newton-Schulz 5 (Keller Jordan, github.com/KellerJordan/Muon)
- Polar Express (Amsel, Persson, Musco, Gower, arXiv:2505.16932;
    github.com/thinking-machines-lab/manifolds)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg

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


### Polynomial sign iteration ###############################################


@dataclass(frozen=True)
class MSignConfig:
  """Configuration for polynomial matrix sign iteration.

  coeffs: per-step (a, b, c) for the quintic p(S) = aI + bS + cS^2.
           If fewer entries than steps, the last entry is repeated.
  steps: number of iterations.
  norm_eps: additive epsilon for normalization: X / (||X|| * norm_scale + norm_eps).
  norm_scale: multiplicative scale for normalization.
  horner: evaluation form for p(S)X where p(S) = aI + bS + cS^2, S = XX^T.
           False ("direct"): a*X + (b*S + c*S@S) @ X  (Muon/NS5 original, avoids identity)
           True ("horner"):  ((c*S + b*I) @ S + a*I) @ X  (Polar Express original)
           Both are algebraically identical but give different bf16 rounding.
  traced: if True, track per-step singular values and off-diagonal energy.
  """
  coeffs: tuple[tuple[float, float, float], ...]
  steps: int = 5
  norm_eps: float = 0.0
  norm_scale: float = 1.0
  horner: bool = False
  traced: bool = False


NS5_COEFFS = (_NS5_ABC,)
POLAR_COEFFS = tuple(_POLAR_EXPRESS_ABC_STABLE)


@jax.jit(static_argnames=("config",))
def msign(G, config, U=None, V=None):
  """Unified polynomial matrix sign iteration.

  Computes approximate polar factor of G in bfloat16 via the quintic
  polynomial iteration X_{k+1} = (a_k I + b_k S + c_k S^2) X_k
  where S = X_k X_k^T.

  When config.traced=True, U and V (precomputed fp32 SVD bases of G)
  must be provided for per-step telemetry.

  Returns (X, aux) where aux is {} when not traced, or
  {"sigmas": (steps+1, min_dim), "offdiags": (steps+1,)} when traced.
  """
  assert G.ndim >= 2
  X = G.astype(jnp.bfloat16)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT

  norm = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
  X = X / (norm * config.norm_scale + config.norm_eps)

  if config.horner:
    I = jnp.eye(X.shape[-2], dtype=X.dtype)

  if config.traced:
    L, R = (V, U) if transposed else (U, V)
    min_dim = min(G.shape[-2], G.shape[-1])
    sigmas = jnp.zeros((config.steps + 1, min_dim))
    offdiags = jnp.zeros(config.steps + 1)
    d, od = _extract_trace(X, L, R)
    sigmas = sigmas.at[0].set(d)
    offdiags = offdiags.at[0].set(od)

  for step in range(config.steps):
    idx = min(step, len(config.coeffs) - 1)
    a, b, c = config.coeffs[idx]
    S = X @ X.mT
    if config.horner:
      Y = c * S + b * I
      Y = Y @ S + a * I
      X = Y @ X
    else:
      B = b * S + c * S @ S
      X = a * X + B @ X

    if config.traced:
      d, od = _extract_trace(X, L, R)
      sigmas = sigmas.at[step + 1].set(d)
      offdiags = offdiags.at[step + 1].set(od)

  if transposed:
    X = X.mT

  if config.traced:
    return X, {"sigmas": sigmas, "offdiags": offdiags}
  return X, {}


### Newton polar iteration ##################################################


def _extract_trace(X, L, R):
  """Extract diagonal and off-diagonal energy of L^T X R."""
  M = L.mT @ X.astype(jnp.float32) @ R
  diag = jnp.diag(M)
  offdiag = jnp.linalg.norm(M - jnp.diag(diag))
  return diag, offdiag


def _newton_step(X):
  """Scaled Newton step for polar decomposition. Cubic convergence."""
  m = X.shape[-2]
  gram = X @ X.mT
  gamma = jnp.exp(-jnp.linalg.slogdet(gram)[1] / (2 * m))
  return (gamma * X + jnp.linalg.solve(gram, X) / gamma) / 2


@jax.jit(static_argnames=("steps",))
def newton_polar(G, steps=10):
  """Scaled Newton iteration for polar decomposition via matrix solve.

  BLAS3 baseline: X_{k+1} = (γ X_k + γ^{-1} (X_k X_k^T)^{-1} X_k) / 2,
  where γ = det(X_k X_k^T)^{-1/(2m)}.
  Cubic convergence. Runs in fp32.
  """
  assert G.ndim >= 2
  X = G.astype(jnp.float32)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT

  X = X / jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)

  for _ in range(steps):
    X = _newton_step(X)

  if transposed:
    X = X.mT
  return X


@jax.jit(static_argnames=("steps",))
def newton_polar_traced(G, U, V, steps=10):
  """Scaled Newton polar iteration with per-step tracking."""
  assert G.ndim == 2
  X = G.astype(jnp.float32)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT
    L, R = V, U
  else:
    L, R = U, V
  min_dim = min(G.shape)

  X = X / jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)

  sigmas = jnp.zeros((steps + 1, min_dim))
  offdiags = jnp.zeros(steps + 1)
  d, od = _extract_trace(X, L, R)
  sigmas = sigmas.at[0].set(d)
  offdiags = offdiags.at[0].set(od)

  for i in range(steps):
    X = _newton_step(X)
    d, od = _extract_trace(X, L, R)
    sigmas = sigmas.at[i + 1].set(d)
    offdiags = offdiags.at[i + 1].set(od)

  if transposed:
    X = X.mT
  return X, sigmas, offdiags


### QR baselines ############################################################


def _signed_qr(X):
  """QR with positive-diagonal R (unique Q)."""
  Q, R = jnp.linalg.qr(X)
  signs = jnp.sign(jnp.diag(R))
  return Q * signs


@jax.jit
def householder_qr(G):
  """Q factor via Householder QR. Single-shot, no iterations.

  Returns Q (same shape as G) in fp32, with positive-diagonal R convention.
  """
  assert G.ndim >= 2
  return _signed_qr(G.astype(jnp.float32))


@jax.jit
def householder_qr_bf16(G):
  """Q factor via Householder QR on bf16 input.

  Casts input to bf16 before QR (JAX may upcast internally for LAPACK).
  Returns Q in fp32, with positive-diagonal R convention.
  """
  assert G.ndim >= 2
  return _signed_qr(G.astype(jnp.bfloat16).astype(jnp.float32))


@jax.jit
def cholesky_qr(G):
  """Q factor via Cholesky QR: R = chol(G^T G), Q = G R^{-1}. Single-shot.

  Less stable than Householder for ill-conditioned G, but pure BLAS3.
  Cholesky R has positive diagonal by construction.
  Returns Q (same shape as G) in fp32.
  """
  assert G.ndim >= 2
  X = G.astype(jnp.float32)
  transposed = G.shape[-2] < G.shape[-1]
  if transposed:
    X = X.mT

  R = jnp.linalg.cholesky(X.mT @ X, upper=True)
  # Q = X R^{-1}: solve R^T Q^T = X^T (lower triangular system)
  Q = jax.scipy.linalg.solve_triangular(R.mT, X.mT, lower=True).mT

  if transposed:
    Q = Q.mT
  return Q
