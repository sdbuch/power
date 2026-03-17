import jax


def main():
  jax.distributed.initialize()

  print("hi")
