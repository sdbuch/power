from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro
from jax.experimental.multihost_utils import process_allgather

from power.msign import newtonschulz5, polar_express


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

  if args.experiment in ("ns5", "both"):
    result = newtonschulz5(G, steps=args.steps_ns5).astype(jnp.float32)
    err = jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)
    if args.m >= args.n:
      orth_err = jnp.linalg.norm(
        result.mT @ result - jnp.eye(min_dim)
      ) / jnp.linalg.norm(jnp.eye(min_dim))
    else:
      orth_err = jnp.linalg.norm(
        result @ result.mT - jnp.eye(min_dim)
      ) / jnp.linalg.norm(jnp.eye(min_dim))
    all_err = process_allgather(jnp.array([err]))
    all_orth = process_allgather(jnp.array([orth_err]))
    if jax.process_index() == 0:
      print(f"NS5 (steps={args.steps_ns5}, n_hosts={len(all_err)}):")
      print(f"  mean relative error vs SVD: {jnp.mean(all_err):.6f}")
      print(f"  mean orthogonality error:   {jnp.mean(all_orth):.6f}")

  if args.experiment in ("polar", "both"):
    result = polar_express(G, steps=args.steps_polar)
    err = jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)
    if args.m >= args.n:
      orth_err = jnp.linalg.norm(
        result.mT @ result - jnp.eye(min_dim)
      ) / jnp.linalg.norm(jnp.eye(min_dim))
    else:
      orth_err = jnp.linalg.norm(
        result @ result.mT - jnp.eye(min_dim)
      ) / jnp.linalg.norm(jnp.eye(min_dim))
    all_err = process_allgather(jnp.array([err]))
    all_orth = process_allgather(jnp.array([orth_err]))
    if jax.process_index() == 0:
      print(f"Polar Express (steps={args.steps_polar}, n_hosts={len(all_err)}):")
      print(f"  mean relative error vs SVD: {jnp.mean(all_err):.6f}")
      print(f"  mean orthogonality error:   {jnp.mean(all_orth):.6f}")


def main():
  jax.distributed.initialize()
  args = tyro.cli(Args)
  test_msign(args)
