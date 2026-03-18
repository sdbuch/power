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
  (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in _POLAR_EXPRESS_ABC[:-1]
] + [_POLAR_EXPRESS_ABC[-1]]


### Helpers ##################################################################


def _extract_trace(X, L, R):
  """Extract diagonal and off-diagonal energy of L^T X R."""
  M = L.mT @ X.astype(jnp.float32) @ R
  diag = jnp.diag(M)
  offdiag = jnp.linalg.norm(M - jnp.diag(diag))
  return diag, offdiag


def _trace_init(X, L, R, steps, min_dim):
  """Allocate and fill step-0 trace arrays."""
  sigmas = jnp.zeros((steps + 1, min_dim))
  offdiags = jnp.zeros(steps + 1)
  d, od = _extract_trace(X, L, R)
  return sigmas.at[0].set(d), offdiags.at[0].set(od)


def _trace_step(X, L, R, sigmas, offdiags, step):
  """Record trace arrays at a given step."""
  d, od = _extract_trace(X, L, R)
  return sigmas.at[step + 1].set(d), offdiags.at[step + 1].set(od)


def _setup_transpose(G, U, V, dtype):
  """Cast G, transpose if tall, select SVD bases L, R accordingly."""
  X = G.astype(dtype)
  transposed = G.shape[-2] > G.shape[-1]
  if transposed:
    X = X.mT
    L, R = V, U
  else:
    L, R = U, V
  return X, L, R, transposed


### Polynomial sign iteration ###############################################


NS5_COEFFS = (_NS5_ABC,)
POLAR_COEFFS = tuple(_POLAR_EXPRESS_ABC_STABLE)


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


jax.tree_util.register_dataclass(
  MSignConfig,
  data_fields=[],
  meta_fields=[
    "coeffs",
    "steps",
    "norm_eps",
    "norm_scale",
    "horner",
    "traced",
  ],
)


@jax.jit
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
    sigmas, offdiags = _trace_init(X, L, R, config.steps, min_dim)

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
      sigmas, offdiags = _trace_step(X, L, R, sigmas, offdiags, step)

  if transposed:
    X = X.mT

  if config.traced:
    return X, {"sigmas": sigmas, "offdiags": offdiags}
  return X, {}


### Newton polar iteration ##################################################


@dataclass(frozen=True)
class NewtonPolarConfig:
  """Configuration for scaled Newton polar iteration.

  steps: number of iterations.
  traced: if True, track per-step singular values and off-diagonal energy.
  """

  steps: int = 10
  traced: bool = False


jax.tree_util.register_dataclass(
  NewtonPolarConfig,
  data_fields=[],
  meta_fields=["steps", "traced"],
)


def _newton_step(X):
  """Scaled Newton step for polar decomposition. Cubic convergence."""
  m = X.shape[-2]
  gram = X @ X.mT
  gamma = jnp.exp(-jnp.linalg.slogdet(gram)[1] / (2 * m))
  return (gamma * X + jnp.linalg.solve(gram, X) / gamma) / 2


@jax.jit
def newton_polar(G, config, U=None, V=None):
  """Scaled Newton iteration for polar decomposition via matrix solve.

  BLAS3 baseline: X_{k+1} = (γ X_k + γ^{-1} (X_k X_k^T)^{-1} X_k) / 2,
  where γ = det(X_k X_k^T)^{-1/(2m)}.
  Cubic convergence. Runs in fp32.

  Returns (X, aux), same convention as msign.
  """
  assert G.ndim >= 2
  X, L, R, transposed = _setup_transpose(G, U, V, jnp.float32)
  min_dim = min(G.shape[-2], G.shape[-1])

  X = X / jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)

  if config.traced:
    sigmas, offdiags = _trace_init(X, L, R, config.steps, min_dim)

  for i in range(config.steps):
    X = _newton_step(X)
    if config.traced:
      sigmas, offdiags = _trace_step(X, L, R, sigmas, offdiags, i)

  if transposed:
    X = X.mT

  if config.traced:
    return X, {"sigmas": sigmas, "offdiags": offdiags}
  return X, {}


### QR baselines ############################################################


@dataclass(frozen=True)
class QRConfig:
  """Configuration for QR-based orthogonalization.

  method: "householder" or "cholesky".
  dtype: computation dtype ("float32" or "bfloat16").
  """

  method: str = "householder"
  dtype: str = "float32"


jax.tree_util.register_dataclass(
  QRConfig,
  data_fields=[],
  meta_fields=["method", "dtype"],
)

_DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


def _signed_qr(X):
  """QR with positive-diagonal R (unique Q)."""
  Q, R = jnp.linalg.qr(X)
  signs = jnp.sign(jnp.diag(R))
  return Q * signs


@jax.jit
def qr(G, config):
  """Q factor via QR decomposition.

  method="householder": Householder QR (LAPACK). LAPACK requires fp32+,
    so bf16 dtype means: cast input to bf16 (quantize), upcast to fp32 for QR.
  method="cholesky": Cholesky QR (R = chol(G^T G), Q = G R^{-1}).
    Less stable for ill-conditioned G, but pure BLAS3.
    Cholesky R has positive diagonal by construction.
    With bf16 dtype, the Gram matrix X^T X is computed in bf16;
    Cholesky and triangular solve run in fp32.

  Returns Q (same shape as G).
  """
  assert G.ndim >= 2
  dtype = _DTYPE_MAP[config.dtype]

  if config.method == "householder":
    # LAPACK doesn't support bf16; quantize input then upcast for QR.
    X = G.astype(dtype).astype(jnp.float32)
    return _signed_qr(X)

  elif config.method == "cholesky":
    X = G.astype(dtype)
    transposed = G.shape[-2] < G.shape[-1]
    if transposed:
      X = X.mT

    # Gram matrix computed in config.dtype (bf16 matmul if requested).
    gram = X.mT @ X
    # Cholesky and solve in fp32 (LAPACK requirement).
    R = jnp.linalg.cholesky(gram.astype(jnp.float32), upper=True)
    Q = jax.scipy.linalg.solve_triangular(
      R.mT,
      X.astype(jnp.float32).mT,
      lower=True,
    ).mT

    if transposed:
      Q = Q.mT
    return Q

  else:
    raise ValueError(f"Unknown QR method: {config.method}")
