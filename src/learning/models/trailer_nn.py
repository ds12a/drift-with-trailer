from flax import nnx

class TrailerModel(nnx.Module):
    def __init__(self, in_dim, out_dim):
        rng = nnx.Rngs(1248)

        self.model = nnx.Sequential(
            nnx.Linear(in_dim, 32, rngs=rng),
            nnx.relu,
            nnx.Linear(32, 32, rngs=rng),
            nnx.relu,
            nnx.Linear(32, out_dim, rngs=rng),
        )

    def __call__(self, x):
        return self.model(x) 
    