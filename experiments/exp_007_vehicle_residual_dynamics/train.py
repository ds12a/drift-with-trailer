import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from flax import nnx
from src.learning.datasets.trailer_data import DataStore, DataLoader
from src.learning.models.trailer_spec import KIN_FS
from src.learning.models.trailer_spec_nores import RAW_FS
from src.learning.models.trailer_nn import TrailerModel
import wandb
# import pickle
import orbax.checkpoint as ocp
# from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs

# from src.simulation.config.trailer_bicycle_config import (
#     TrailerBicycleEnvConfig,
#     VehicleConfig,
#     TrackConfig,
#     SimulationConfig,
# )

class ChannelLoss(nnx.Metric):
    def __init__(self, num_channels, argname="channel_losses"):
        self.n = num_channels
        self.argname = argname

        self.total = nnx.metrics.MetricState(jnp.zeros(self.n, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.zeros(self.n, dtype=jnp.int32))

    def update(self, **kwargs):
        loss = kwargs[self.argname]
        self.count.value += loss.shape[0]
        self.total.value += jnp.sum(loss, axis=0)
    
    def compute(self):
        return self.total.value / jnp.maximum(self.count.value, 1) # no div by zero
    
    def reset(self):
        self.total.value = jnp.zeros_like(self.total.value)
        self.count.value = jnp.zeros_like(self.count.value)


@nnx.jit
def loss_fn(model, batch):
    x, y = batch
    return ((model(x) - y) ** 2).mean()
@nnx.jit
def col_loss(model, batch):
    x, y = batch
    return ((model(x) - y) ** 2)

@nnx.jit
def train_step(model, optimizer, metrics, batch):
    cl = col_loss(model, batch)
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    metrics.update(loss=loss, channel_losses=cl)
    return loss

# @nnx.jit
def eval_step(model, state):
    return model(state[None, ...])[0]

CHANNELS = ("ax", "ay", "w1", "w2")

class LearnedDynamics:
    def __init__(
        self,
        model,
        data: DataLoader,
        optimizer_params={"learning_rate": 1e-3},
        iodims=(6, 4),
        batch_size=4096,
        key=jax.random.PRNGKey(0),
    ):
        self.idim, self.odim = iodims
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.key = key
        self.optimizer = nnx.Optimizer(self.model, optax.adam(**optimizer_params), wrt=nnx.Param)
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            channel_losses=ChannelLoss(self.odim),
        )
        self.loss_history = []
        self.test_loss_history = []
        self.y_std = np.asarray(data.y_std)

    def train(self, epochs, checkpoint_freq=5):
        best = None
        for e in range(epochs):
            train, test = self.data.get_data(
                self.batch_size, jax.random.fold_in(self.key, e)
            )
            for i, batch in enumerate(train):
                train_step(self.model, self.optimizer, self.metrics, batch)

            self.loss_history.append(self.metrics.compute())
            self.metrics.reset()

            for i, batch in enumerate(test):
                self.metrics.update(loss=loss_fn(self.model, batch),
                                    channel_losses=col_loss(self.model, batch))

            self.test_loss_history.append(self.metrics.compute())
            self.metrics.reset()

            tr, te = self.loss_history[-1], self.test_loss_history[-1]
            tl, vl = float(tr["loss"]), float(te["loss"])

            raw_rmse = np.sqrt(np.asarray(te["channel_losses"]) * self.y_std**2)

            wandb.log({f"test_rmse_raw/{c}": float(r)
                    for c, r in zip(CHANNELS, raw_rmse)}, step=e)

            print("   raw RMSE  " + "  ".join(
                f"{c}:{r:.4f}" for c, r in zip(CHANNELS, raw_rmse)))

            wandb.log({
                "epoch": e,
                "train/loss": tl,
                "test/loss": vl,
                "test/rmse": np.sqrt(vl),
                **{f"test_rmse_raw/{c}": float(r) for c, r in zip(CHANNELS, raw_rmse)},
                **{f"train/{c}": float(v) for c, v in zip(CHANNELS, tr["channel_losses"])},
                **{f"test/{c}": float(v) for c, v in zip(CHANNELS, te["channel_losses"])},
            }, step=e)

            print(f"\rEpoch {e}\t Train loss: {tl:.5f}\tTest loss: {vl:.5f}"
                  f"\tTest RMSE: {np.sqrt(vl):.5f}\t"
                  + " ".join(f"{c}:{v:.3f}" for c, v in zip(CHANNELS, te["channel_losses"])) + 
                  "\traw RMSE: " + "  ".join(
                    f"{c}:{r:.4f}" for c, r in zip(CHANNELS, raw_rmse)))

            if e > 0 and e % checkpoint_freq == 0:
                self.save(output="src/learning/models/trained/trailer-nokin")
                if best is None or vl < best:
                    best = vl
                    wandb.run.summary["best_test_loss"] = vl
                    wandb.run.summary["best_epoch"] = e
                    self.save(output="src/learning/models/trained/trailer-nokin-best")

    # def _unnormalize(self, dynamics):
    #     return dynamics * self.dynamics_std + self.dynamics_mean
    
    def save(self, output="src/learning/models/trained/trailer"):
        graphdef, state = nnx.split(self.model)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(Path.cwd() / output, state, force=True)
        checkpointer.wait_until_finished()

    def load(self, source="src/learning/models/trained/trailer"):
        graphdef, state = nnx.split(self.model)
        checkpointer = ocp.StandardCheckpointer()
        restored_state = checkpointer.restore(source, state)
        nnx.update(self.model, restored_state)

if __name__ == "__main__":

    spec = RAW_FS

    raw = DataStore.load(Path("./experiments/exp_007_vehicle_residual_dynamics/data_raw.npz"))
    data = raw.build(spec, True)

    wandb.init(
        project="Train",
        config={
            "learning_rate": 1e-2,
            "batch_size": 4096,
            "H": spec.H,
            "F": spec.F,
            "data_version": spec.data_version,
            "split_seed": spec.split_seed,
            "train_frac": spec.train_frac,
            "n_train": len(data.train),
            "n_test": len(data.test),
            "y_std": data.y_std.tolist(),
        },
    )

    learned = LearnedDynamics(
        TrailerModel(spec.H * 6, 4), data,
        {"learning_rate": wandb.config.learning_rate},  # not wandb.config directly
        batch_size=wandb.config.batch_size,
    )
    learned.train(500)
    learned.save()
    data.save(Path("./experiments/exp_007_vehicle_residual_dynamics/data_proc1.npz"))