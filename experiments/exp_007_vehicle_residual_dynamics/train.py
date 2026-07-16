import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from flax import nnx
from src.learning.datasets.trailer_data import Data
from src.learning.models.trailer_nn import TrailerModel
import pickle
import orbax.checkpoint as ocp
from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs

from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)


def dynamics(obs):
    sh, ch, vx, vy, truck_yaw_rate, yaw_trailer_rate, mu, delta, accel = obs

    vehicle = params.vehicle
    hitch = jnp.arctan2(sh, ch) 

    L1 = vehicle.lf + vehicle.lr
    L2 = vehicle.l2f + vehicle.l2r

    theta1_dot = (vx / L1) * jnp.tan(delta)
    theta2_dot = (vx / L2) * (
        jnp.sin(hitch)
        - ((vehicle.hitch_offset - vehicle.lr) / L1) * jnp.cos(hitch) * jnp.tan(delta)
    )
    hitch_dot = theta1_dot - theta2_dot

    throttle = jnp.maximum(accel, 0.0)
    brake = -jnp.minimum(accel, 0.0)
    commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake

    return jnp.array(
        [
            ch * hitch_dot,
            -sh * hitch_dot,
            commanded + vy * truck_yaw_rate,
            0.0,
            0.0,
            0.0,
        ]
    )


pred_batch = jax.jit(jax.vmap(dynamics))


@nnx.jit
def loss_fn(model, batch, dyn_mean, dyn_std, state_mean, state_std):
    x, y = batch

    true_y = y * dyn_std + dyn_mean
    last_x = x[:, -9:] # TODO for safety, hardcoding is probably bad here
    true_x = last_x * state_std + state_mean

    pred = pred_batch(true_x.reshape(-1, 9)).reshape(y.shape)
    true_res = true_y - pred
    norm_res = (true_res - dyn_mean) / dyn_std
    return ((model(x) - norm_res) ** 2).mean()


@nnx.jit
def train_step(model, optimizer, metrics, batch, dyn_mean, dyn_std, state_mean, state_std):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, dyn_mean, dyn_std, state_mean, state_std)
    optimizer.update(model, grads)
    metrics.update(loss=loss)

    return loss


# @nnx.jit
def eval_step(model, state):
    return model(state[None, ...])[0]


class LearnedDynamics:
    def __init__(
        self,
        model,
        batch_size,
        state_mean,
        state_std,
        dynamics_mean,
        dynamics_std,
        optimizer_params={"learning_rate": 1e-3},
    ):
        self.model = model
        self.state_std = state_std
        self.state_mean = state_mean
        self.dynamics_std = dynamics_std
        self.dynamics_mean = dynamics_mean
        self.data = Data(batch_size, state_mean, state_std, dynamics_mean, dynamics_std)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(**optimizer_params), wrt=nnx.Param)
        self.metrics = nnx.metrics.Average("loss")
        self.loss_history = []
        self.test_loss_history = []

    def __call__(self, state, action):
        full_state = jnp.concatenate([state, action])
        norm = (full_state - self.state_mean) / self.state_std
        return dynamics(full_state) + self._unnormalize(self.model(norm))

    def train(self, epochs, checkpoint_freq=5):

        best = None

        for e in range(epochs):
            train, test = self.data.get_data()

            for i, batch in enumerate(train):
                # print(f"\rTrain Batch {i}", end="")
                train_step(self.model, self.optimizer, self.metrics, batch, self.data.dynamics_mean, self.data.dynamics_std, self.data.state_mean, self.data.state_std)
            
            self.loss_history.append(self.metrics.compute())
            self.metrics.reset()

            for i, batch in enumerate(test):
                # print(f"\rTrain done, Test Batch {i}", end="")
                self.metrics.update(loss=loss_fn(self.model, batch, self.data.dynamics_mean, self.data.dynamics_std, self.data.state_mean, self.data.state_std))

            self.test_loss_history.append(self.metrics.compute())
            self.metrics.reset()

            print(f"\rEpoch {e}\t Train loss: {self.loss_history[-1]}\tTest loss: {self.test_loss_history[-1]}\tTest RMSE: {np.sqrt(self.test_loss_history[-1])}")

            if e > 0 and e % checkpoint_freq == 0:
                self.save()

                if best is None or self.test_loss_history[-1] < best:
                    self.save(output="src/learning/models/trained/trailer-best")



    def _unnormalize(self, dynamics):
        return dynamics * self.dynamics_std + self.dynamics_mean
    
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

params = TrailerBicycleEnvConfig("scenario", TrackConfig(), VehicleConfig(), SimulationConfig())



if __name__ == "__main__":

    learned = LearnedDynamics(
        TrailerModel(36, 6), 4096, 
        np.array([0, 0, 10, 0, 0, 0, 0.8, 0, 0]), 
        np.array([0.4, 0.2, 17.3, 1.5, 0.35, 0.35, 0.1, 0.4, 0.4]), 
        np.zeros(6), 
        np.array([0.4, 0.4, 1.5, 1.5, 0.8, 0.8]),
    )

    with open("./experiments/exp_007_vehicle_residual_dynamics/data.pkl", "rb") as f:
        learned.data = pickle.load(f) 

    print("Opened pickle")
    print(f"Dataset has {len(learned.data)} samples")
    learned.data.batch_size = 4096

    learned.train(500)

    learned.save()