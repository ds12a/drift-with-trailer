from flax import nnx


class TrailerModel(nnx.Module):
    def __init__(self, in_dim, out_dim):
        rng = nnx.Rngs(1248)

        self.model = nnx.Sequential(
            nnx.Linear(in_dim, 512, rngs=rng),
            nnx.silu,
            nnx.Linear(512, 512, rngs=rng),
            nnx.silu,
            # nnx.Linear(256, 256, rngs=rng),
            # nnx.silu,
            # nnx.Linear(256, 256, rngs=rng),
            # nnx.silu,
            nnx.Linear(512, out_dim, rngs=rng),
        )

    def __call__(self, x):
        return self.model(x)
