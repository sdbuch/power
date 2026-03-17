from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import tyro
from jax.experimental.multihost_utils import process_allgather

from power.msign import newtonschulz5, polar_express

EXPERIMENTS = {
    "ns5": ("NS5", newtonschulz5),
    "polar": ("Polar Express", polar_express),
}


@dataclass
class Args:
  experiment: str = "both"  # "ns5", "polar", or "both"
  m: int = 512
  n: int = 256
  steps_ns5: int = 5
  steps_polar: int = 5
  seed: int = 42


def svd_sign(G):
  """Ground truth polar factor via SVD."""
  U, _, Vt = jnp.linalg.svd(G, full_matrices=False)
  return U @ Vt


def test_msign(args):
  local_seed = args.seed + jax.process_index()
  key = jax.random.key(local_seed)
  G = jax.random.normal(key, (args.m, args.n))
  truth = svd_sign(G)
  min_dim = min(args.m, args.n)

  def evaluate(name, fn, G, truth, min_dim):
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
    name, fn = EXPERIMENTS[key]
    evaluate(
        f"{name} (steps={steps[key]})",
        partial(fn, steps=steps[key]),
        G, truth, min_dim,
    )


def main():
  jax.distributed.initialize()
  args = tyro.cli(Args)
  test_msign(args)
