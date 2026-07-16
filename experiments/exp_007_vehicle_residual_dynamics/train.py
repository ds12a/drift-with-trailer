import jax
import jax.numpy as jnp
import optax
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
def loss_fn(model, batch):
    x, y = batch
    pred = pred_batch(x.reshape(-1, 9)).reshape(y.shape)
    return ((model(x) - (y - pred)) ** 2).mean()


@nnx.jit
def train_step(model, optimizer, metrics, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
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
        optimizer_params={"learning_rate": 0.05},
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

    def train(self, epochs, checkpoint_freq=50):
        for i in range(epochs):
            train, test = self.data.get_data()

            for batch in zip(*train):
                train_step(self.model, self.optimizer, self.metrics, batch)

            self.loss_history.append(self.metrics.compute())
            self.metrics.reset()

            for batch in zip(*test):
                self.metrics.update(loss=loss_fn(self.model, batch))

            self.test_loss_history.append(self.metrics.compute())
            self.metrics.reset()

            print(f"Epoch {i}\tTrain: {self.loss_history[-1]}\tTest: {self.test_loss_history[-1]}")

            if i > 0 and i % checkpoint_freq == 0:
                self.save()

    def _unnormalize(self, dynamics):
        return dynamics * self.dynamics_std + self.dynamics_mean

    def save(self, output="src/learning/models/trained/trailer.pkl"):
        with open(output, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, source="src/learning/models/trained/trailer.pkl"):
        with open(source, "rb") as f:
            self.model = pickle.load(f)


params = TrailerBicycleEnvConfig("scenario", TrackConfig(), VehicleConfig(), SimulationConfig())


learned = LearnedDynamics(
    TrailerModel(9, 6), 64, jnp.zeros(9), jnp.ones(9), jnp.zeros(6), jnp.ones(6)
)

learned.data = pickle.load()  # TODO

learned.train(1000)
