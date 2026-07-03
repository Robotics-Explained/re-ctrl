"""skrl model/agent builders matching the tau-ctrl / SB3 setup ([256,256] MLP,
lr 3e-4, batch 256, gamma .99, polyak .005). Kept separate so the benchmark
stays readable. Mirrors skrl's official SAC/TD3 example structure."""

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class GaussianActor(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
                               min_log_std=-20, max_log_std=2)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)
        self.register_buffer("act_scale", high)

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        # tanh-bound the mean to the action range. skrl's GaussianMixin has no
        # squashing, so with only env-side clipping the critic is evaluated on
        # unbounded actions and the actor mean runs away to +/-1e6. Bounding the
        # mean in-network is the standard skrl pattern for bounded-action envs.
        mean = torch.tanh(self.mean_layer(x)) * self.act_scale
        return mean, self.log_std_parameter, {}


class DetActor(DeterministicMixin, Model):
    """Bounded deterministic actor (TD3): tanh-scaled to the action range."""

    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.num_actions), nn.Tanh(),
        )
        high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)
        self.register_buffer("scale", high)

    def compute(self, inputs, role):
        return self.net(inputs["states"]) * self.scale, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


def build_sac(env, device, warmup):
    from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    obs, act = env.observation_space, env.action_space
    models = {
        "policy": GaussianActor(obs, act, device),
        "critic_1": Critic(obs, act, device),
        "critic_2": Critic(obs, act, device),
        "target_critic_1": Critic(obs, act, device),
        "target_critic_2": Critic(obs, act, device),
    }
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update(dict(
        batch_size=256, discount_factor=0.99, polyak=0.005,
        actor_learning_rate=3e-4, critic_learning_rate=3e-4, entropy_learning_rate=3e-4,
        learn_entropy=True, gradient_steps=1, learning_starts=warmup, random_timesteps=warmup,
    ))
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    mem = RandomMemory(memory_size=100_000, num_envs=env.num_envs, device=device)
    return SAC(models=models, memory=mem, cfg=cfg,
               observation_space=obs, action_space=act, device=device)


def build_td3(env, device, warmup):
    from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.resources.noises.torch import GaussianNoise

    obs, act = env.observation_space, env.action_space
    models = {
        "policy": DetActor(obs, act, device),
        "target_policy": DetActor(obs, act, device),
        "critic_1": Critic(obs, act, device),
        "critic_2": Critic(obs, act, device),
        "target_critic_1": Critic(obs, act, device),
        "target_critic_2": Critic(obs, act, device),
    }
    cfg = TD3_DEFAULT_CONFIG.copy()
    cfg.update(dict(
        batch_size=256, discount_factor=0.99, polyak=0.005,
        actor_learning_rate=3e-4, critic_learning_rate=3e-4,
        gradient_steps=1, learning_starts=warmup, random_timesteps=warmup,
        policy_delay=2,
        # Exploration noise (was None → TD3 never explored → failed). Match
        # tau-ctrl/SB3's TD3 defaults: expl_noise 0.1, target smoothing 0.2/0.5.
        exploration={"noise": GaussianNoise(0.0, 0.1, device=device),
                     "initial_scale": 1.0, "final_scale": 1.0, "timesteps": None},
        smooth_regularization_noise=GaussianNoise(0.0, 0.2, device=device),
        smooth_regularization_clip=0.5,
    ))
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    mem = RandomMemory(memory_size=100_000, num_envs=env.num_envs, device=device)
    return TD3(models=models, memory=mem, cfg=cfg,
               observation_space=obs, action_space=act, device=device)
