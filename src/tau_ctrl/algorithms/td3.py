"""Twin Delayed DDPG (TD3) — off-policy, continuous-action RL.

Deterministic policy + twin Q-critics (like SAC) plus two extra tricks that
fix DDPG's instability: target policy smoothing (noise on the target action
so the critic can't be exploited by sharp Q-peaks) and delayed policy
updates (the actor and target networks update less often than the critics).
Canonical recipe from Fujimoto et al. 2018.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseController, ControllerManifest, register
from .off_policy import (
    ReplayBuffer,
    build_mlp,
    hard_update,
    random_action,
    run_vectorized_training,
    soft_update,
)


def _torch():
    import torch  # noqa: PLC0415
    return torch


@register("td3")
class TD3(BaseController):
    manifest = ControllerManifest(
        name="td3", family="rl", trainable=True, gpu_capable=True,
    )

    def __init__(
        self,
        env: Any,
        hidden: tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        expl_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        **kw: Any,
    ) -> None:
        self.hidden = tuple(hidden)
        self.lr, self.gamma, self.tau = lr, gamma, tau
        self.buffer_size = int(buffer_size)
        self.expl_noise, self.target_noise, self.noise_clip = expl_noise, target_noise, noise_clip
        self.policy_delay = int(policy_delay)
        super().__init__(env, **kw)

    def _setup(self) -> None:
        torch = _torch()
        import torch.nn as nn

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.obs_dim, self.act_dim = obs_dim, act_dim

        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self._act_scale = torch.as_tensor((high - low) / 2.0, device=self.device)
        self._act_bias = torch.as_tensor((high + low) / 2.0, device=self.device)
        self._act_low = torch.as_tensor(low, device=self.device)
        self._act_high = torch.as_tensor(high, device=self.device)

        self.actor = build_mlp(nn, (obs_dim, *self.hidden, act_dim), nn.ReLU).to(self.device)
        self.actor_targ = build_mlp(nn, (obs_dim, *self.hidden, act_dim), nn.ReLU).to(self.device)
        hard_update(self.actor_targ, self.actor)

        def make_q():
            return build_mlp(nn, (obs_dim + act_dim, *self.hidden, 1), nn.ReLU).to(self.device)

        self.q1, self.q2 = make_q(), make_q()
        self.q1_targ, self.q2_targ = make_q(), make_q()
        hard_update(self.q1_targ, self.q1)
        hard_update(self.q2_targ, self.q2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr
        )
        # GPU-resident buffer: sample() returns device tensors, zero copies per update.
        self._buffer = ReplayBuffer(self.buffer_size, obs_dim, act_dim, device=self.device)
        # Pre-allocate obs tensor for predict() — avoids a GPU alloc on every call.
        self._obs_in = torch.zeros(1, obs_dim, dtype=torch.float32, device=self.device)
        self._update_count = 0

    # ------------------------------------------------------------------
    def _actor_forward(self, obs_t, torch, target: bool = False):
        net = self.actor_targ if target else self.actor
        return torch.tanh(net(obs_t)) * self._act_scale + self._act_bias

    def predict(self, obs, state=None, deterministic: bool = True):
        torch = _torch()
        # Re-use pre-allocated GPU tensor — avoids a GPU allocation per call.
        self._obs_in.copy_(torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)))
        with torch.no_grad():
            action = self._actor_forward(self._obs_in, torch, target=False)
            if not deterministic:
                action = action + torch.randn_like(action) * self.expl_noise * self._act_scale
        return self._clip_action(action.cpu().numpy().ravel()), None

    # ------------------------------------------------------------------
    def _vec_action(self, obs, warmup: bool, torch):
        """Batched action for the vectorized loop — no numpy, stays on device."""
        if warmup:
            u = torch.rand(obs.shape[0], self.act_dim, device=self.device) * 2.0 - 1.0
            return u * self._act_scale + self._act_bias
        with torch.no_grad():
            action = self._actor_forward(obs, torch, target=False)
            action = action + torch.randn_like(action) * self.expl_noise * self._act_scale
            action = torch.clamp(action, self._act_low, self._act_high)
        return action

    def learn(self, total_timesteps: int = 10_000, warmup_steps: int = 1000,
              batch_size: int = 256, updates_per_step: int = 1, **kw: Any) -> "TD3":
        torch = _torch()
        env = self.env
        # Vectorized on-device env (has num_envs) → the GPU-saturating fast path.
        if getattr(env, "num_envs", None):
            return run_vectorized_training(
                self, total_timesteps, warmup_steps, batch_size, updates_per_step, torch
            )
        obs, _ = env.reset()
        for t in range(int(total_timesteps)):
            if t < warmup_steps:
                action = random_action(self.action_space, self.rng)
            else:
                action, _ = self.predict(obs, deterministic=False)
            next_obs, reward, term, trunc, _ = env.step(action)
            self._buffer.add(obs, action, reward, next_obs, term)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
            if len(self._buffer) >= batch_size and t >= warmup_steps:
                self._update(batch_size, torch)
        return self

    def _update(self, batch_size: int, torch) -> None:
        # sample() returns GPU tensors directly — no copies, no allocations.
        batch    = self._buffer.sample(batch_size)
        obs      = batch["obs"]
        act      = batch["act"]
        rew      = batch["reward"]    # shape (B, 1) already
        next_obs = batch["next_obs"]
        done     = batch["done"]      # shape (B, 1) already

        with torch.no_grad():
            next_action = self._actor_forward(next_obs, torch, target=True)
            # Canonical TD3: noise is N(0, target_noise) in normalised [-1,1] space,
            # scaled to action space; clip to [-noise_clip, noise_clip] in action units.
            noise = (torch.randn_like(next_action) * self.target_noise * self._act_scale).clamp(
                -self.noise_clip * self._act_scale, self.noise_clip * self._act_scale
            )
            next_action = torch.clamp(next_action + noise, self._act_low, self._act_high)
            # Compute next_obs-action pair once — used by both target Q networks.
            next_sa = torch.cat([next_obs, next_action], -1)
            q_targ = torch.min(
                self.q1_targ(next_sa),
                self.q2_targ(next_sa),
            )
            target = rew + self.gamma * (1.0 - done) * q_targ

        # Compute obs-action pair once — used by both live Q networks.
        sa = torch.cat([obs, act], -1)
        q1_pred = self.q1(sa)
        q2_pred = self.q2(sa)
        q_loss = 0.5 * (((q1_pred - target) ** 2).mean() + ((q2_pred - target) ** 2).mean())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            actor_action = self._actor_forward(obs, torch, target=False)
            actor_loss = -self.q1(torch.cat([obs, actor_action], -1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor_targ, self.actor, self.tau)
            soft_update(self.q1_targ, self.q1, self.tau)
            soft_update(self.q2_targ, self.q2, self.tau)

    # ------------------------------------------------------------------
    def _state_dict(self):
        return {
            "actor": self.actor.state_dict(), "actor_targ": self.actor_targ.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_targ": self.q1_targ.state_dict(), "q2_targ": self.q2_targ.state_dict(),
        }

    def _load_state_dict(self, sd):
        if not sd:
            return
        self.actor.load_state_dict(sd["actor"])
        self.actor_targ.load_state_dict(sd["actor_targ"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.q1_targ.load_state_dict(sd["q1_targ"])
        self.q2_targ.load_state_dict(sd["q2_targ"])
