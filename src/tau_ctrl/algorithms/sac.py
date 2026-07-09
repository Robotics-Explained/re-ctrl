"""Soft Actor-Critic (SAC) — off-policy, continuous-action RL.

Unlike PPO's on-policy rollouts (discard data after each update), SAC learns
from a replay buffer of all past transitions, which makes it far more
sample-efficient for continuous robot control — the standard workhorse for
real-robot RL where environment steps are expensive.

Squashed-Gaussian policy (tanh-bounded), twin Q-critics (reduces
overestimation bias), automatic entropy-temperature tuning — the canonical
SAC recipe (Haarnoja et al. 2018), following the well-tested spinningup
formulation for the tanh log-prob correction.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import (
    BaseController,
    ControllerManifest,
    register,
    require_box_action_space,
    require_torch,
)
from .off_policy import (
    ReplayBuffer,
    build_mlp,
    hard_update,
    random_action,
    run_vectorized_training,
    soft_update,
)


def _torch():
    return require_torch()


@register("sac")
class SAC(BaseController):
    manifest = ControllerManifest(
        name="sac", family="rl", trainable=True, gpu_capable=True,
    )

    def __init__(
        self,
        env: Any,
        hidden: tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        target_entropy: Optional[float] = None,
        **kw: Any,
    ) -> None:
        self.hidden = tuple(hidden)
        self.lr, self.gamma, self.tau = lr, gamma, tau
        self.buffer_size = int(buffer_size)
        self._target_entropy_override = target_entropy
        super().__init__(env, **kw)

    def _setup(self) -> None:
        require_box_action_space(self.action_space, "SAC")
        torch = _torch()
        import torch.nn as nn

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.obs_dim, self.act_dim = obs_dim, act_dim

        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self._act_scale = torch.as_tensor((high - low) / 2.0, device=self.device)
        self._act_bias = torch.as_tensor((high + low) / 2.0, device=self.device)

        # Actor: shared trunk, separate mean / log-std heads (squashed Gaussian).
        self.actor_trunk = build_mlp(nn, (obs_dim, *self.hidden), nn.ReLU, nn.ReLU).to(self.device)
        self.mu_head = nn.Linear(self.hidden[-1], act_dim).to(self.device)
        self.log_std_head = nn.Linear(self.hidden[-1], act_dim).to(self.device)

        # Twin Q-critics + targets.
        def make_q():
            return build_mlp(nn, (obs_dim + act_dim, *self.hidden, 1), nn.ReLU).to(self.device)

        self.q1, self.q2 = make_q(), make_q()
        self.q1_targ, self.q2_targ = make_q(), make_q()
        hard_update(self.q1_targ, self.q1)
        hard_update(self.q2_targ, self.q2)

        actor_params = (
            list(self.actor_trunk.parameters())
            + list(self.mu_head.parameters())
            + list(self.log_std_head.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr
        )

        self.target_entropy = (
            self._target_entropy_override
            if self._target_entropy_override is not None
            else -float(act_dim)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # GPU-resident buffer: sample() returns device tensors, zero copies per update.
        self._buffer = ReplayBuffer(self.buffer_size, obs_dim, act_dim, device=self.device)
        # Pre-allocate obs tensor for predict() — avoids a GPU alloc on every call.
        self._obs_in = torch.zeros(1, obs_dim, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def _actor_forward(self, obs_t, deterministic: bool, torch):
        h = self.actor_trunk(obs_t)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-20.0, 2.0)
        std = torch.exp(log_std)
        pre_tanh = mu if deterministic else mu + std * torch.randn_like(mu)
        y = torch.tanh(pre_tanh)
        action = y * self._act_scale + self._act_bias

        normal = torch.distributions.Normal(mu, std)
        logp = normal.log_prob(pre_tanh).sum(-1)
        # Stable tanh-squash correction: log(1 - tanh(x)^2) = 2*(log2 - x - softplus(-2x))
        logp = logp - (
            2.0 * (np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        ).sum(-1)
        return action, logp

    def predict(self, obs, state=None, deterministic: bool = True):
        torch = _torch()
        # Re-use pre-allocated GPU tensor — avoids a GPU allocation per call.
        self._obs_in.copy_(torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)))
        with torch.no_grad():
            action, _ = self._actor_forward(self._obs_in, deterministic, torch)
        return self._clip_action(action.cpu().numpy().ravel()), None

    # ------------------------------------------------------------------
    def _vec_action(self, obs, warmup: bool, torch):
        """Batched action for the vectorized loop — no numpy, stays on device."""
        if warmup:
            u = torch.rand(obs.shape[0], self.act_dim, device=self.device) * 2.0 - 1.0
            return u * self._act_scale + self._act_bias
        with torch.no_grad():
            action, _ = self._actor_forward(obs, False, torch)
        return action

    def learn(self, total_timesteps: int = 10_000, warmup_steps: int = 1000,
              batch_size: int = 256, updates_per_step: int = 1, **kw: Any) -> "SAC":
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
        alpha    = self.log_alpha.exp()

        with torch.no_grad():
            next_action, next_logp = self._actor_forward(next_obs, False, torch)
            next_logp = next_logp.unsqueeze(-1)
            # Compute next_obs-action pair once — reused by both target Q networks.
            next_sa = torch.cat([next_obs, next_action], -1)
            q_targ = torch.min(
                self.q1_targ(next_sa),
                self.q2_targ(next_sa),
            )
            target = rew + self.gamma * (1.0 - done) * (q_targ - alpha * next_logp)

        q1_pred = self.q1(torch.cat([obs, act], -1))
        q2_pred = self.q2(torch.cat([obs, act], -1))
        q_loss = 0.5 * (((q1_pred - target) ** 2).mean() + ((q2_pred - target) ** 2).mean())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        pi_action, logp_pi = self._actor_forward(obs, False, torch)
        logp_pi = logp_pi.unsqueeze(-1)
        q_pi = torch.min(self.q1(torch.cat([obs, pi_action], -1)), self.q2(torch.cat([obs, pi_action], -1)))
        actor_loss = (alpha.detach() * logp_pi - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (logp_pi.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.q1_targ, self.q1, self.tau)
        soft_update(self.q2_targ, self.q2, self.tau)

    # ------------------------------------------------------------------
    def _state_dict(self):
        return {
            "actor_trunk": self.actor_trunk.state_dict(),
            "mu_head": self.mu_head.state_dict(),
            "log_std_head": self.log_std_head.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_targ": self.q1_targ.state_dict(), "q2_targ": self.q2_targ.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
        }

    def _load_state_dict(self, sd):
        if not sd:
            return
        torch = _torch()
        self.actor_trunk.load_state_dict(sd["actor_trunk"])
        self.mu_head.load_state_dict(sd["mu_head"])
        self.log_std_head.load_state_dict(sd["log_std_head"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.q1_targ.load_state_dict(sd["q1_targ"])
        self.q2_targ.load_state_dict(sd["q2_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(sd["log_alpha"], device=self.device))
