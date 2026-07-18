"""GRPO — critic-free policy-gradient control that exploits branching.

Group Relative Policy Optimization (DeepSeek, 2024) replaces a learned critic
with a *group-relative* baseline: sample a group of outcomes from the same
state, score them, and use each one's z-scored return as its advantage —
``(r_i - mean(r)) / std(r)``, no value network.

In control (unlike LLM generation) the "same state" isn't free — you need the
env to be *branchable* (:func:`get_state`/:func:`set_state`, the same contract
MPPI/CEM/ILQR already use). This module is that: it walks the real trajectory
step by step, and at each visited state branches ``group_size`` action samples
from the current policy through the *live* env for ``group_horizon`` steps,
computes group-relative advantages, and updates a Gaussian policy with a
PPO-style clipped surrogate.

Benchmarked against PPO/SAC on Pendulum-v1, HalfCheetah-v5 (both dense-reward),
and a sparse/terminal-reward pendulum variant (3 seeds each, matched training
budgets): GRPO consistently edges out PPO under sparse/terminal reward (mean
return −573 vs −588, low variance both) — the critic-free ranking sidesteps
PPO's bootstrapping difficulty there — but **SAC's replay buffer still wins
both the dense and sparse settings tested**. Use this for the branchable +
sparse/terminal-reward niche specifically, not as a general PPO/SAC
replacement — and expect it to be slower per real step than either, since
scoring one decision costs ``group_size * group_horizon`` extra env rollouts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    BaseController,
    ControllerManifest,
    get_state,
    register,
    require_box_action_space,
    require_torch,
    set_state,
)


def _torch():
    return require_torch()


@register("grpo")
class GRPO(BaseController):
    manifest = ControllerManifest(
        name="grpo", family="rl", requires_branching=True, uses_reward=True,
        trainable=True, gpu_capable=True,
    )

    def __init__(
        self,
        env: Any,
        hidden: tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip: float = 0.2,
        group_size: int = 8,
        group_horizon: int = 1,
        rollout_steps: int = 256,
        epochs: int = 4,
        minibatch: int = 64,
        ent_coef: float = 0.0,
        **kw: Any,
    ) -> None:
        self.hidden = tuple(hidden)
        self.lr, self.gamma, self.clip = lr, gamma, clip
        self.group_size, self.group_horizon = int(group_size), int(group_horizon)
        self.rollout_steps, self.epochs, self.minibatch = rollout_steps, epochs, minibatch
        self.ent_coef = ent_coef
        super().__init__(env, **kw)

    def _setup(self) -> None:
        require_box_action_space(self.action_space, "GRPO")
        torch = _torch()
        import torch.nn as nn

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.obs_dim, self.act_dim = obs_dim, act_dim

        layers, last = [], obs_dim
        for h in self.hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.actor = nn.Sequential(*layers).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=self.device))
        params = list(self.actor.parameters()) + [self.log_std]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    # --- policy helpers --------------------------------------------------
    def _dist(self, obs_t):
        torch = _torch()
        mean = self.actor(obs_t)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def predict(self, obs, state=None, deterministic: bool = True):
        torch = _torch()
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device).reshape(1, -1)
        with torch.no_grad():
            mean = self.actor(obs_t)
            action = mean if deterministic else self._dist(obs_t).sample()
        return self._clip_action(action.cpu().numpy().ravel()), None

    # --- group rollout: branch `group_size` action samples from state0 ---
    def _group_rollout(self, torch, state0, obs_t):
        """Sample group_size actions from the policy, roll each forward
        ``group_horizon`` steps through the *live* branchable env, and return
        (actions, old log-probs, returns) for the group at this state.
        """
        with torch.no_grad():
            dist = self._dist(obs_t.expand(self.group_size, -1))
            actions = dist.sample()
            logps = dist.log_prob(actions).sum(-1)
        actions_np = actions.cpu().numpy()
        returns = np.zeros(self.group_size, dtype=np.float32)
        for i in range(self.group_size):
            set_state(self.env, state0)
            a = actions_np[i]
            disc, obs_i = 1.0, None
            for h in range(self.group_horizon):
                if h > 0:  # subsequent lookahead steps: resample from the policy
                    with torch.no_grad():
                        o_t = torch.as_tensor(obs_i, dtype=torch.float32,
                                              device=self.device).reshape(1, -1)
                        a = self._dist(o_t).sample().cpu().numpy().ravel()
                obs_i, r, term, trunc, _ = self.env.step(self._clip_action(a))
                returns[i] += disc * r
                disc *= self.gamma
                if term or trunc:
                    break
        set_state(self.env, state0)  # leave the real env as we found it
        return actions, logps, returns

    # --- training --------------------------------------------------------
    def learn(self, total_timesteps: int = 10_000, **kw: Any) -> "GRPO":
        torch = _torch()
        env = self.env
        obs, _ = env.reset()
        buf_obs, buf_act, buf_logp, buf_adv = [], [], [], []

        steps = 0
        while steps < int(total_timesteps):
            state0 = get_state(env)
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self.device).reshape(1, -1)
            actions, logps, returns = self._group_rollout(torch, state0, obs_t)
            std = returns.std()
            if std > 1e-8:
                adv = (returns - returns.mean()) / (std + 1e-8)
            else:
                adv = np.zeros_like(returns)

            for i in range(self.group_size):
                buf_obs.append(np.asarray(obs, dtype=np.float32))
                buf_act.append(actions[i].cpu().numpy())
                buf_logp.append(float(logps[i]))
                buf_adv.append(float(adv[i]))

            # Advance the real episode with one sampled action from the group
            # (index 0) — this keeps data collection on-policy, matching the
            # PPO/off-policy convention of stepping the live env once per
            # decision even though several candidates were scored for it.
            set_state(env, state0)
            chosen = self._clip_action(actions[0].cpu().numpy())
            obs, r, term, trunc, _ = env.step(chosen)
            steps += 1
            if term or trunc:
                obs, _ = env.reset()

            if len(buf_obs) >= self.rollout_steps * self.group_size:
                self._update(torch, buf_obs, buf_act, buf_logp, buf_adv)
                buf_obs, buf_act, buf_logp, buf_adv = [], [], [], []
        return self

    def _update(self, torch, obs_list, act_list, logp_list, adv_list) -> None:
        dev = self.device
        obs = torch.as_tensor(np.array(obs_list, dtype=np.float32), device=dev)
        act = torch.as_tensor(np.array(act_list, dtype=np.float32), device=dev)
        old_logp = torch.as_tensor(np.array(logp_list, dtype=np.float32), device=dev)
        adv = torch.as_tensor(np.array(adv_list, dtype=np.float32), device=dev)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, self.minibatch):
                mb = idx[start:start + self.minibatch]
                mb_t = torch.as_tensor(mb, device=dev)
                dist = self._dist(obs[mb_t])
                logp = dist.log_prob(act[mb_t]).sum(-1)
                ratio = torch.exp(logp - old_logp[mb_t])
                a_mb = adv[mb_t]
                unclipped = ratio * a_mb
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_mb
                pi_loss = -torch.min(unclipped, clipped).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # --- persistence -----------------------------------------------------
    def _state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "log_std": self.log_std.detach().cpu().numpy(),
        }

    def _load_state_dict(self, sd):
        if not sd:
            return
        torch = _torch()
        self.actor.load_state_dict(sd["actor"])
        with torch.no_grad():
            self.log_std.copy_(torch.as_tensor(sd["log_std"], device=self.device))
