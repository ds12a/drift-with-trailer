import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from flax import nnx
from src.learning.datasets.trailer_data import DataStore, DataLoader#, DataLoaderBlocked
    
from src.learning.models.trailer_nn import TrailerModel
import wandb

from src.learning.models.beamng_trailer_spec import STATE_FS as SPEC, IN_COLS

# from src.learning.models.beamng_model_spec import STATE_FS as SPEC, IN_COLS

# RES = True

# if RES:
#     from src.learning.models.trailer_spec import KIN_FS as SPEC, IN_COLS
# else:
#     from src.learning.models.trailer_spec_nores import RAW_FS as SPEC, IN_COLS

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
        return self.total.value / jnp.maximum(self.count.value, 1)  # no div by zero

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
    return (model(x) - y) ** 2


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


CHANNELS = ("ax", "ay", "alpha1", "alpha2", "steer", "accel")


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

    def train(self, epochs, checkpoint_freq=1):
        best = None
        best_epoch = 0
        for e in range(epochs):
            train, test = self.data.get_data(self.batch_size, jax.random.fold_in(self.key, e))
            for i, batch in enumerate(train):
                train_step(self.model, self.optimizer, self.metrics, batch)

            self.loss_history.append(self.metrics.compute())
            self.metrics.reset()

            for i, batch in enumerate(test):
                self.metrics.update(
                    loss=loss_fn(self.model, batch), channel_losses=col_loss(self.model, batch)
                )

            self.test_loss_history.append(self.metrics.compute())
            self.metrics.reset()

            tr, te = self.loss_history[-1], self.test_loss_history[-1]
            tl, vl = float(tr["loss"]), float(te["loss"])

            raw_rmse = np.sqrt(np.asarray(te["channel_losses"]) * self.y_std**2)

            wandb.log({f"test_rmse_raw/{c}": float(r) for c, r in zip(CHANNELS, raw_rmse)}, step=e)

            # print("   raw RMSE  " + "  ".join(f"{c}:{r:.4f}" for c, r in zip(CHANNELS, raw_rmse)))

            wandb.log(
                {
                    "epoch": e,
                    "progression/best_rmse": best,
                    "progression/best_epoch": best_epoch,
                    "train/loss": tl,
                    "test/loss": vl,
                    "test/rmse": np.sqrt(vl),
                    **{f"test_rmse_raw/{c}": float(r) for c, r in zip(CHANNELS, raw_rmse)},
                    **{f"train/{c}": float(v) for c, v in zip(CHANNELS, tr["channel_losses"])},
                    **{f"test/{c}": float(v) for c, v in zip(CHANNELS, te["channel_losses"])},
                },
                step=e,
            )

            print(
                f"\rEpoch {e}\t Train loss: {tl:.5f}\tTest loss: {vl:.5f}"
                # f"\tTest RMSE: {np.sqrt(vl):.5f}\t"
                # + " ".join(f"{c}:{v:.3f}" for c, v in zip(CHANNELS, te["channel_losses"]))
                # + "\traw RMSE: "
                # + "  ".join(f"{c}:{r:.4f}" for c, r in zip(CHANNELS, raw_rmse))
                + f"\tbest epoch: {best_epoch}, eval: {best}"
            )

            if e > 0 and e % checkpoint_freq == 0:
                self.save(output=f"src/learning/models/trained/{wandb.config.run_id}")
                if best is None or vl < best:
                    best = vl
                    best_epoch = e
                    wandb.run.summary["best_test_loss"] = vl
                    wandb.run.summary["best_epoch"] = e
                    self.save(output=f"src/learning/models/trained/{wandb.config.run_id}_best")

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

    def ax_floor(self, mu_col=6, a_col=8, chan=0):
        """Bin ax residual (raw units) by mu x |throttle|. Prints std + count/1k per cell.
        Reveals whether the ax floor is friction-saturation (grows w/ |a|, worse low mu)
        or a fit/capacity issue (flat everywhere)."""
        spec, W = self.data.spec, np.arange(self.data.spec.H + self.data.spec.F)
        ys = np.asarray(self.data.y_std)[chan]

        res, mu, a = [], [], []
        for i in range(0, len(self.data.test), 4096):
            idx = self.data.test[i : i + 4096]
            w = self.data.data[idx[:, None] + W]                       # (B, H+F, D)
            x = (spec.encode_x(w) - self.data.x_mean) / self.data.x_std
            y = (spec.encode_y(w) - self.data.y_mean) / self.data.y_std
            res.append(np.asarray(self.model(x)[:, chan] - y[:, chan]) * ys)  # raw units
            k = w[:, spec.H - 1]                                        # row t
            mu.append(np.asarray(k[:, mu_col])); a.append(np.abs(np.asarray(k[:, a_col])))
        res, mu, a = map(np.concatenate, (res, mu, a))

        print(f"\nax residual std (raw m/s^2), by mu x |throttle|:")
        for m in np.unique(mu):
            cells = []
            for lo, hi in [(0, 0.2), (0.2, 0.6), (0.6, 1.01)]:
                s = (mu == m) & (a >= lo) & (a < hi)
                cells.append(f"{res[s].std():.2f}({s.sum()//1000}k)" if s.sum() else "  --  ")
            print(f"  mu={m:.1f}   |a|<.2 {cells[0]:>10}   .2-.6 {cells[1]:>10}   >.6 {cells[2]:>10}")
        print(f"  overall: {res.std():.3f}   (grows w/ |a| & low mu => friction floor; "
              f"flat => fit issue)")

if __name__ == "__main__":

    NPZ_SAVE_HEAD = "data_proc_test4"

    spec = SPEC   

    raw = DataStore.load(Path("./experiments/exp_008_beamng/data_trial2_aug.npz"))
    data: DataLoader = raw.build(spec, DataLoader)

    wandb.init(
        project="Train_beamng",
        config={
            "learning_rate":1e-3,
            "batch_size": 256,
            "H": spec.H,
            "F": spec.F,
            "data_version": spec.data_version,
            "split_seed": spec.split_seed,
            "train_frac": spec.train_frac,
            "n_train": len(data.train),
            "n_test": len(data.test),
            "y_std": data.y_std.tolist(),
            "run_id": "beamng-l4-128-test4"
        },
    )

    learned = LearnedDynamics(
        TrailerModel(spec.H * len(IN_COLS), 6), data,
        {"learning_rate": wandb.config.learning_rate},  # not wandb.config directly
        batch_size=wandb.config.batch_size,
        iodims=(len(IN_COLS), 6)
    )
    # learned.load(Path.cwd() / "src/learning/models/trained/trailer-kin-512-best")
    try:
        learned.train(250)
    # learned.save()
    finally:
        learned.ax_floor()
        data.save(Path(f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}.npz"))
        data.norm_to_json(Path(f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}_stats.json"))
