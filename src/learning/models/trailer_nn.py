from flax import nnx


class TrailerModel(nnx.Module):
    def __init__(self, in_dim, out_dim, total=None):
        rng = nnx.Rngs(1248)

        if total is None:
            self.model = nnx.Sequential(
                nnx.Linear(in_dim, 128, rngs=rng),
                nnx.silu,
                nnx.Linear(128, 128, rngs=rng),
                nnx.silu,
                nnx.Linear(128, 128, rngs=rng),
                nnx.silu,
                nnx.Linear(128, 128, rngs=rng),
                nnx.silu,
                nnx.Linear(128, out_dim, rngs=rng),
            )

        else:
            arr = []
            for i in range(len(total) - 2):
                arr.append(nnx.Linear(total[i], total[i + 1], rngs=rng))
                arr.append(nnx.silu)
            arr.append(nnx.Linear(total[-2], total[-1], rngs=rng))
            self.model = nnx.Sequential(*arr)

    def __call__(self, x):
        return self.model(x)
