from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import tyro
from jax.experimental.multihost_utils import process_allgather

from power.msign import (
  cholesky_qr,
  householder_qr,
  householder_qr_bf16,
  newton_polar,
  newton_polar_traced,
  newtonschulz5,
  newtonschulz5_traced,
  polar_express,
  polar_express_traced,
)

# (name, fn, fn_traced, truth_key)
EXPERIMENTS = {
  "ns5": ("NS5", newtonschulz5, newtonschulz5_traced, "polar"),
  "polar": ("Polar Express", polar_express, polar_express_traced, "polar"),
  "newton": ("Newton Polar (fp32)", newton_polar, newton_polar_traced, "polar"),
  "householder": ("Householder QR (fp32)", householder_qr, None, "qr"),
  "householder_bf16": ("Householder QR (bf16)", householder_qr_bf16, None, "qr"),
  "cholesky": ("Cholesky QR (fp32)", cholesky_qr, None, "qr"),
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


def test_msign(args):
  local_seed = args.seed + jax.process_index()
  key = jax.random.key(local_seed)
  G = make_matrix(args.matrix, args.m, args.n, key)
  truths, sigma, U, V = make_truths(G)
  min_dim = min(G.shape[-2], G.shape[-1])

  if jax.process_index() == 0:
    print(f"matrix={args.matrix}, shape={G.shape}, kappa={sigma[0] / sigma[-1]:.1f}")

  def evaluate(name, fn, fn_traced, truth_key, G, U, V, min_dim):
    truth = truths[truth_key]
    if jax.process_index() == 0:
      print(f"{name} (n_hosts={jax.process_count()}):")
    if args.trace and fn_traced is not None:
      result, sigmas, offdiags = fn_traced(G, U, V)
      result = result.astype(jnp.float32)
      if jax.process_index() == 0:
        print("  singular value evolution (min, median, max) | offdiag norm:")
        for step in range(sigmas.shape[0]):
          s = sigmas[step]
          print(
            f"    step {step}: "
            f"min={jnp.min(s):.6f}  "
            f"med={jnp.median(s):.6f}  "
            f"max={jnp.max(s):.6f}  "
            f"| offdiag={offdiags[step]:.6f}"
          )
    else:
      result = fn(G).astype(jnp.float32)

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

  steps = {"ns5": args.steps_ns5, "polar": args.steps_polar, "newton": args.steps_newton}
  keys = GROUPS.get(args.experiment, [args.experiment])
  for key in keys:
    name, fn, fn_traced, truth_key = EXPERIMENTS[key]
    if key in steps:
      fn = partial(fn, steps=steps[key])
      if fn_traced is not None:
        fn_traced = partial(fn_traced, steps=steps[key])
      name = f"{name} (steps={steps[key]})"
    evaluate(name, fn, fn_traced, truth_key, G, U, V, min_dim)


def main():
  try:
    jax.distributed.initialize()
  except ValueError:
    pass
  args = tyro.cli(Args)
  test_msign(args)
