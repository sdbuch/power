from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro

from power.msign import newtonschulz5, polar_express


@dataclass
class Args:
  experiment: str = "both"  # "ns5", "polar", or "both"
  m: int = 512
  n: int = 256
  steps_ns5: int = 5
  steps_polar: int = 10
  seed: int = 42


def svd_sign(G):
  """Ground truth polar factor via SVD."""
  U, _, Vt = jnp.linalg.svd(G, full_matrices=False)
  return U @ Vt


def test_msign(args):
  key = jax.random.key(args.seed)
  G = jax.random.normal(key, (args.m, args.n))
  truth = svd_sign(G)

  min_dim = min(args.m, args.n)

  if args.experiment in ("ns5", "both"):
    result = newtonschulz5(G, steps=args.steps_ns5).astype(jnp.float32)
    err = jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)
    if args.m >= args.n:
      orth_err = jnp.linalg.norm(result.mT @ result - jnp.eye(min_dim))
    else:
      orth_err = jnp.linalg.norm(result @ result.mT - jnp.eye(min_dim))
    print(f"NS5 (steps={args.steps_ns5}):")
    print(f"  relative error vs SVD: {err:.6f}")
    print(f"  orthogonality error:   {orth_err:.6f}")

  if args.experiment in ("polar", "both"):
    result = polar_express(G, steps=args.steps_polar)
    err = jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)
    if args.m >= args.n:
      orth_err = jnp.linalg.norm(result.mT @ result - jnp.eye(min_dim))
    else:
      orth_err = jnp.linalg.norm(result @ result.mT - jnp.eye(min_dim))
    print(f"Polar Express (steps={args.steps_polar}):")
    print(f"  relative error vs SVD: {err:.6f}")
    print(f"  orthogonality error:   {orth_err:.6f}")


def main():
  jax.distributed.initialize()
  args = tyro.cli(Args)
  test_msign(args)
