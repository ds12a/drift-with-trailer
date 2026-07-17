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


# Full kin state is (x, y, phi1, phi2, vx, vy) -> full kin time deriv is (vx, vy, phidot1, phidot2, ax, ay)
# Kin doesnt really do lat accel

# NN + kin will provide (in long, lat framing) (phidot1, phi2dot, ax, ay)
# Dyn data is  d/dt (sin (hitch), cos(hitch), vx, vy, truck yaw, trailer yaw)

# def dynamics(obs):
    # sh, ch, vx, vy, truck_yaw_rate, yaw_trailer_rate, mu, delta, accel = obs

    # u = jnp.array([delta, accel])
    # # hitch = jnp.atan2(sh, ch)

    # vehicle = params.vehicle
    # steer_cmd = jnp.clip(u[0], -1.0, 1.0)
    # accel_cmd = jnp.clip(u[1], -1.0, 1.0)

    # throttle = jnp.maximum(accel_cmd, 0.0)
    # brake = -jnp.minimum(accel_cmd, 0.0)

    # commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake

    # dt = params.simulation.dt

    # throttle = jnp.maximum(accel_cmd, 0.0)
    # brake = -jnp.minimum(accel_cmd, 0.0)

    # delta = steer_cmd * vehicle.max_steer_rad

    # theta1_dot = (vx / (vehicle.lf + vehicle.lr)) * jnp.tan(delta)
    # theta2_dot = (vx / (vehicle.l2f + vehicle.l2r)) * (
    #     sh
    #     - ((vehicle.hitch_offset - vehicle.lr) / (vehicle.lf + vehicle.lr))
    #     * ch
    #     * jnp.tan(delta)
    # )

    # return jnp.array([theta1_dot, theta2_dot, commanded, 0])


# pred_batch = jax.jit(jax.vmap(dynamics))


@nnx.jit
def loss_fn(model, batch, dyn_mean, dyn_std, state_mean, state_std):
    x, y = batch

    true_y = y * dyn_std + dyn_mean
    last_x = x[:, -9:] # TODO for safety, hardcoding is probably bad here
    true_x = last_x * state_std + state_mean

    # pred = pred_batch(true_x.reshape(-1, 9)).reshape(y.shape)
    # true_res = true_y - pred
    true_res = true_y
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
        self.data: Data = Data(batch_size, state_mean, state_std, dynamics_mean, dynamics_std)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(**optimizer_params), wrt=nnx.Param)
        self.metrics = nnx.metrics.Average("loss")
        self.loss_history = []
        self.test_loss_history = []



    # def __call__(self, state, action):
    #     full_state = jnp.concatenate([state, action])
    #     norm = (full_state - self.state_mean) / self.state_std
    #     return dynamics(full_state) + self._unnormalize(self.model(norm))

    def train(self, epochs, checkpoint_freq=5):

        best = None

        # train, test = self.data.get_data()
        # fixed_batch = next(iter(train))          # hoisted out
        # for e in range(epochs):
        #     loss = train_step(self.model, self.optimizer, self.metrics,
        #                     fixed_batch, self.data.dynamics_mean, self.data.dynamics_std, self.data.state_mean, self.data.state_std)
        #     print(f"Epoch {e}\t{self.metrics.compute()}")
        #     self.metrics.reset()

        for e in range(epochs):
            train, test = self.data.get_data()

            for i, batch in enumerate(train):
                # print(f"\rTrain Batch {i}", end="")
                train_step(self.model, self.optimizer, self.metrics, batch, self.data.dynamics_mean, self.data.dynamics_std, self.data.state_mean, self.data.state_std)
                break
            
            self.loss_history.append(self.metrics.compute())
            self.metrics.reset()

            for i, batch in enumerate(test):
                # print(f"\rTrain done, Test Batch {i}", end="")
                self.metrics.update(loss=loss_fn(self.model, batch, self.data.dynamics_mean, self.data.dynamics_std, self.data.state_mean, self.data.state_std))
                break

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

    with open("./experiments/exp_007_vehicle_residual_dynamics/data_compiled.pkl", "rb") as f:
        learned.data = pickle.load(f) 

    print("Opened pickle")
    print(f"Dataset has {len(learned.data)} samples")

    learned.dynamics_mean = learned.data.dynamics_mean
    learned.dynamics_std = learned.data.dynamics_std
    learned.state_mean = learned.data.state_mean
    learned.state_std = learned.data.state_std

    learned.data.batch_size = 4096

    learned.train(50)

    learned.save()