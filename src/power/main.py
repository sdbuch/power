from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import tyro
from jax.experimental.multihost_utils import process_allgather

from power.msign import (
  MSIgnConfig,
  NS5_COEFFS,
  POLAR_COEFFS,
  cholesky_qr,
  householder_qr,
  householder_qr_bf16,
  msign,
  newton_polar,
  newton_polar_traced,
  # Legacy functions for verification.
  newtonschulz5,
  newtonschulz5_traced,
  polar_express,
  polar_express_traced,
)

# Default configs for the polynomial sign iterations.
NS5 = MSIgnConfig(coeffs=NS5_COEFFS, norm_eps=1e-7)
POLAR = MSIgnConfig(coeffs=POLAR_COEFFS, norm_scale=1.01, horner=True)


def _msign_entry(name, default_config, truth_key):
  """Build an EXPERIMENTS entry for a polynomial sign iteration."""
  return (name, default_config, truth_key)


# (name, config_or_fn, truth_key)
# For msign-based experiments, config_or_fn is an MSIgnConfig.
# For standalone experiments, config_or_fn is (fn, fn_traced) or (fn, None).
EXPERIMENTS = {
  "ns5": ("NS5", NS5, "polar"),
  "polar": ("Polar Express", POLAR, "polar"),
  "newton": ("Newton Polar (fp32)", (newton_polar, newton_polar_traced), "polar"),
  "householder": ("Householder QR (fp32)", (householder_qr, None), "qr"),
  "householder_bf16": ("Householder QR (bf16)", (householder_qr_bf16, None), "qr"),
  "cholesky": ("Cholesky QR (fp32)", (cholesky_qr, None), "qr"),
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
  verify: bool = False  # run legacy functions side-by-side and compare


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
  return {
    "polar": U @ Vt,
    "qr": householder_qr(G),
  }, sigma, U, Vt.mT


# Legacy dispatch for --verify mode.
_LEGACY = {
  "ns5": (newtonschulz5, newtonschulz5_traced),
  "polar": (polar_express, polar_express_traced),
}


def test_msign(args):
  local_seed = args.seed + jax.process_index()
  key = jax.random.key(local_seed)
  G = make_matrix(args.matrix, args.m, args.n, key)
  truths, sigma, U, V = make_truths(G)
  min_dim = min(G.shape[-2], G.shape[-1])

  if jax.process_index() == 0:
    print(f"matrix={args.matrix}, shape={G.shape}, kappa={sigma[0] / sigma[-1]:.1f}")

  steps_map = {"ns5": args.steps_ns5, "polar": args.steps_polar, "newton": args.steps_newton}

  def evaluate(name, key, entry, G, U, V, truth_key, min_dim):
    truth = truths[truth_key]
    config_or_fn = entry
    s = steps_map.get(key)

    if isinstance(config_or_fn, MSIgnConfig):
      # Polynomial sign iteration via unified msign.
      config = MSIgnConfig(
        coeffs=config_or_fn.coeffs,
        steps=s if s is not None else config_or_fn.steps,
        norm_eps=config_or_fn.norm_eps,
        norm_scale=config_or_fn.norm_scale,
        horner=config_or_fn.horner,
        traced=args.trace,
      )
      result, aux = msign(G, config, U, V)
      result = result.astype(jnp.float32)
      label = f"{name} (steps={config.steps})"

      if args.trace and jax.process_index() == 0:
        sigmas, offdiags = aux["sigmas"], aux["offdiags"]
        print(f"{label} (n_hosts={jax.process_count()}):")
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
      elif jax.process_index() == 0:
        print(f"{label} (n_hosts={jax.process_count()}):")

      # Verify against legacy if requested.
      if args.verify and key in _LEGACY:
        legacy_fn, legacy_fn_traced = _LEGACY[key]
        legacy_result = legacy_fn(G, steps=config.steps).astype(jnp.float32)
        diff = jnp.linalg.norm(result - legacy_result)
        if jax.process_index() == 0:
          print(f"  legacy verification diff: {diff:.8f}")

    else:
      # Standalone method (newton, QR).
      fn, fn_traced = config_or_fn
      if s is not None:
        fn = partial(fn, steps=s)
        if fn_traced is not None:
          fn_traced = partial(fn_traced, steps=s)
      label = f"{name} (steps={s})" if s is not None else name

      if args.trace and fn_traced is not None:
        result, sigmas, offdiags = fn_traced(G, U, V)
        result = result.astype(jnp.float32)
        if jax.process_index() == 0:
          print(f"{label} (n_hosts={jax.process_count()}):")
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
      else:
        result = fn(G).astype(jnp.float32)
        if jax.process_index() == 0:
          print(f"{label} (n_hosts={jax.process_count()}):")

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
    name, config_or_fn, truth_key = EXPERIMENTS[key]
    evaluate(name, key, config_or_fn, G, U, V, truth_key, min_dim)


def main():
  try:
    jax.distributed.initialize()
  except ValueError:
    pass
  args = tyro.cli(Args)
  test_msign(args)
