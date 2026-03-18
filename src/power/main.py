from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import tyro
from jax.experimental.multihost_utils import process_allgather

from power.msign import (
    newtonschulz5,
    newtonschulz5_traced,
    polar_express,
    polar_express_traced,
)

EXPERIMENTS = {
    "ns5": ("NS5", newtonschulz5, newtonschulz5_traced),
    "polar": ("Polar Express", polar_express, polar_express_traced),
}


@dataclass
class Args:
  experiment: str = "both"  # "ns5", "polar", or "both"
  matrix: str = "gaussian"  # "gaussian", "square_gaussian", "tridiag"
  m: int = 512
  n: int = 256
  steps_ns5: int = 5
  steps_polar: int = 5
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
        2 * jnp.eye(d)
        + jnp.diag(-jnp.ones(d - 1), -1)
        + jnp.diag(-jnp.ones(d - 1), 1)
    )
  else:
    raise ValueError(f"Unknown matrix type: {matrix_type}")


def test_msign(args):
  local_seed = args.seed + jax.process_index()
  key = jax.random.key(local_seed)
  G = make_matrix(args.matrix, args.m, args.n, key)
  U, sigma, Vt = jnp.linalg.svd(G, full_matrices=False)
  V = Vt.mT
  truth = U @ Vt
  min_dim = min(G.shape[-2], G.shape[-1])

  if jax.process_index() == 0:
    print(f"matrix={args.matrix}, shape={G.shape}, "
          f"kappa={sigma[0] / sigma[-1]:.1f}")

  def evaluate(name, fn, fn_traced, G, U, V, truth, min_dim):
    if args.trace:
      result, sigmas = fn_traced(G, U, V)
      result = result.astype(jnp.float32)
      if jax.process_index() == 0:
        print(f"  singular value evolution (min, median, max):")
        for step in range(sigmas.shape[0]):
          s = sigmas[step]
          print(f"    step {step}: "
                f"min={jnp.min(s):.6f}  "
                f"med={jnp.median(s):.6f}  "
                f"max={jnp.max(s):.6f}")
    else:
      result = fn(G).astype(jnp.float32)

    err = jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)
    gram = result.mT @ result if G.shape[-2] >= G.shape[-1] else result @ result.mT
    orth_err = jnp.linalg.norm(
        gram - jnp.eye(min_dim)
    ) / jnp.linalg.norm(jnp.eye(min_dim))

    all_err = process_allgather(jnp.array([err]))
    all_orth = process_allgather(jnp.array([orth_err]))
    if jax.process_index() == 0:
      print(f"{name} (n_hosts={len(all_err)}):")
      print(f"  mean relative error vs SVD: {jnp.mean(all_err):.6f}")
      print(f"  mean orthogonality error:   {jnp.mean(all_orth):.6f}")

  steps = {"ns5": args.steps_ns5, "polar": args.steps_polar}
  keys = EXPERIMENTS.keys() if args.experiment == "both" else [args.experiment]
  for key in keys:
    name, fn, fn_traced = EXPERIMENTS[key]
    evaluate(
        f"{name} (steps={steps[key]})",
        partial(fn, steps=steps[key]),
        partial(fn_traced, steps=steps[key]),
        G, U, V, truth, min_dim,
    )


def main():
  jax.distributed.initialize()
  args = tyro.cli(Args)
  test_msign(args)
