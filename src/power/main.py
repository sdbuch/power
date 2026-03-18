from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro
from jax.experimental.multihost_utils import process_allgather

from power.msign import (
  NS5_COEFFS,
  POLAR_COEFFS,
  MSignConfig,
  NewtonPolarConfig,
  QRConfig,
  msign,
  newton_polar,
  qr,
)

# Default configs.
NS5 = MSignConfig(coeffs=NS5_COEFFS, norm_eps=1e-7)
POLAR = MSignConfig(coeffs=POLAR_COEFFS, norm_scale=1.01, horner=True)
NEWTON = NewtonPolarConfig()
HOUSEHOLDER_FP32 = QRConfig(method="householder", dtype="float32")
HOUSEHOLDER_BF16 = QRConfig(method="householder", dtype="bfloat16")
CHOLESKY_FP32 = QRConfig(method="cholesky", dtype="float32")
CHOLESKY_BF16 = QRConfig(method="cholesky", dtype="bfloat16")

# (name, config, truth_key)
EXPERIMENTS = {
  "ns5": ("NS5", NS5, "polar"),
  "polar": ("Polar Express", POLAR, "polar"),
  "newton": ("Newton Polar (fp32)", NEWTON, "polar"),
  "householder": ("Householder QR (fp32)", HOUSEHOLDER_FP32, "qr"),
  "householder_bf16": ("Householder QR (bf16)", HOUSEHOLDER_BF16, "qr"),
  "cholesky": ("Cholesky QR (fp32)", CHOLESKY_FP32, "qr"),
  "cholesky_bf16": ("Cholesky QR (bf16)", CHOLESKY_BF16, "qr"),
}

GROUPS = {
  "all": list(EXPERIMENTS.keys()),
  "both": ["ns5", "polar"],
  "qr": ["householder", "householder_bf16", "cholesky"],
  "polar_all": ["ns5", "polar", "newton"],
}


@dataclass
class Args:
  experiment: str = "all"  # key from EXPERIMENTS or GROUPS
  matrix: str = "gaussian"  # "gaussian", "square_gaussian", "tridiag"
  m: int = 512
  n: int = 256
  steps_ns5: int = 5
  steps_polar: int = 5
  steps_newton: int = 10
  seed: int = 42
  trace: bool = False
  profile: str = ""  # xprof trace dir (e.g. /tmp/jax-trace); empty disables


def make_matrix(matrix_type, m, n, key):
  if matrix_type == "gaussian":
    return jax.random.normal(key, (m, n))
  elif matrix_type == "square_gaussian":
    d = min(m, n)
    return jax.random.normal(key, (d, d))
  elif matrix_type == "tridiag":
    d = min(m, n)
    return (
      2 * jnp.eye(d) + jnp.diag(-jnp.ones(d - 1), -1) + jnp.diag(-jnp.ones(d - 1), 1)
    )
  else:
    raise ValueError(f"Unknown matrix type: {matrix_type}")


def make_truths(G):
  """Compute ground truth targets for each method family."""
  U, sigma, Vt = jnp.linalg.svd(G, full_matrices=False)
  return (
    {
      "polar": U @ Vt,
      "qr": qr(G, HOUSEHOLDER_FP32),
    },
    sigma,
    U,
    Vt.mT,
  )


def _dispatch(config, G, U, V, traced, profile_dir=""):
  """Call the right function for a config, return (result, aux).

  If profile_dir is non-empty, wraps the computation in an xprof trace.
  Runs a warmup call first so the profile captures compiled execution.
  """
  if isinstance(config, MSignConfig):
    cfg = MSignConfig(
      coeffs=config.coeffs,
      steps=config.steps,
      norm_eps=config.norm_eps,
      norm_scale=config.norm_scale,
      horner=config.horner,
      traced=traced,
    )
    fn = lambda: msign(G, cfg, U, V)

  elif isinstance(config, NewtonPolarConfig):
    cfg = NewtonPolarConfig(steps=config.steps, traced=traced)
    fn = lambda: newton_polar(G, cfg, U, V)

  elif isinstance(config, QRConfig):
    fn = lambda: (qr(G, config), {})

  else:
    raise ValueError(f"Unknown config type: {type(config)}")

  # Warmup (triggers compilation).
  result, aux = fn()
  jax.block_until_ready(result)

  if profile_dir:
    jax.profiler.start_trace(profile_dir)

  result, aux = fn()
  jax.block_until_ready(result)

  if profile_dir:
    jax.profiler.stop_trace()

  if isinstance(config, (MSignConfig, NewtonPolarConfig)):
    return result.astype(jnp.float32), aux
  return result.astype(jnp.float32), {}


def _format_label(name, config):
  if isinstance(config, (MSignConfig, NewtonPolarConfig)):
    return f"{name} (steps={config.steps})"
  return name


def test_msign(args):
  local_seed = args.seed + jax.process_index()
  key = jax.random.key(local_seed)
  G = make_matrix(args.matrix, args.m, args.n, key)
  truths, sigma, U, V = make_truths(G)
  min_dim = min(G.shape[-2], G.shape[-1])

  if jax.process_index() == 0:
    print(f"matrix={args.matrix}, shape={G.shape}, kappa={sigma[0] / sigma[-1]:.1f}")

  steps_map = {
    "ns5": args.steps_ns5,
    "polar": args.steps_polar,
    "newton": args.steps_newton,
  }

  def evaluate(name, key, config, G, U, V, truth_key, min_dim):
    # Override steps from CLI if applicable.
    if key in steps_map and isinstance(config, (MSignConfig, NewtonPolarConfig)):
      from dataclasses import replace

      config = replace(config, steps=steps_map[key])

    truth = truths[truth_key]
    if args.profile:
      steps_str = (
        f"_s{config.steps}"
        if isinstance(config, (MSignConfig, NewtonPolarConfig))
        else ""
      )
      profile_dir = f"{args.profile}/{args.matrix}_{args.m}x{args.n}/{key}{steps_str}"
    else:
      profile_dir = ""
    result, aux = _dispatch(config, G, U, V, args.trace, profile_dir)
    label = _format_label(name, config)

    if jax.process_index() == 0:
      print(f"{label} (n_hosts={jax.process_count()}):")

    if args.trace and "sigmas" in aux and jax.process_index() == 0:
      sigmas, offdiags = aux["sigmas"], aux["offdiags"]
      print("  singular value evolution (min, median, max) | offdiag norm:")
      for step in range(sigmas.shape[0]):
        sv = sigmas[step]
        print(
          f"    step {step}: "
          f"min={jnp.min(sv):.6f}  "
          f"med={jnp.median(sv):.6f}  "
          f"max={jnp.max(sv):.6f}  "
          f"| offdiag={offdiags[step]:.6f}"
        )

    err = jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)
    gram = result.mT @ result if G.shape[-2] >= G.shape[-1] else result @ result.mT
    orth_err = jnp.linalg.norm(gram - jnp.eye(min_dim)) / jnp.linalg.norm(
      jnp.eye(min_dim)
    )

    all_err = process_allgather(jnp.array([err]))
    all_orth = process_allgather(jnp.array([orth_err]))
    if jax.process_index() == 0:
      print(f"  mean relative error vs {truth_key} truth: {jnp.mean(all_err):.6f}")
      print(f"  mean orthogonality error:                 {jnp.mean(all_orth):.6f}")

  keys = GROUPS.get(args.experiment, [args.experiment])
  for key in keys:
    name, config, truth_key = EXPERIMENTS[key]
    evaluate(name, key, config, G, U, V, truth_key, min_dim)


def main():
  try:
    jax.distributed.initialize()
  except ValueError:
    pass
  args = tyro.cli(Args)
  test_msign(args)
