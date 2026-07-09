"""A compact, self-contained PPO (continuous actions) in PyTorch.

Standard surface (``learn`` / ``predict`` / ``save`` / ``load``) implementation.
Runs on GPU automatically when torch reports CUDA available (``device="auto"``),
else CPU.

Simulator-agnostic: talks only to the Gymnasium env's ``reset``/``step``.
Kept intentionally small (single-env on-policy rollouts, GAE, clipped
surrogate); enough to learn simple continuous-control tasks and to slot into
the same interface as the model-based controllers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    BaseController,
    ControllerManifest,
    register,
    require_box_action_space,
    require_torch,
)


def _torch():
    return require_torch()


@register("ppo")
class PPO(BaseController):
    manifest = ControllerManifest(
        name="ppo", family="rl", trainable=True, gpu_capable=True,
    )

    def __init__(
        self,
        env: Any,
        hidden: tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        rollout_steps: int = 2048,
        epochs: int = 10,
        minibatch: int = 64,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        **kw: Any,
    ) -> None:
        self.hidden = tuple(hidden)
        self.lr, self.gamma, self.gae_lambda, self.clip = lr, gamma, gae_lambda, clip
        self.rollout_steps, self.epochs, self.minibatch = rollout_steps, epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        super().__init__(env, **kw)

    def _setup(self) -> None:
        require_box_action_space(self.action_space, "PPO")
        torch = _torch()
        import torch.nn as nn

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.obs_dim, act = obs_dim, act_dim

        def mlp(out):
            layers, last = [], obs_dim
            for h in self.hidden:
                layers += [nn.Linear(last, h), nn.Tanh()]
                last = h
            layers += [nn.Linear(last, out)]
            return nn.Sequential(*layers)

        self.actor = mlp(act).to(self.device)
        self.critic = mlp(1).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(act, device=self.device))
        params = list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    # --- policy helpers --------------------------------------------------
    def _dist(self, obs_t):
        torch = _torch()
        mean = self.actor(obs_t)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def predict(self, obs, state=None, deterministic: bool = True):
        torch = _torch()
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(1, -1)
        with torch.no_grad():
            mean = self.actor(obs_t)
            if deterministic:
                action = mean
            else:
                action = self._dist(obs_t).sample()
        return self._clip_action(action.cpu().numpy().ravel()), None

    # --- training --------------------------------------------------------
    def learn(self, total_timesteps: int = 10000, **kw: Any) -> "PPO":
        torch = _torch()
        env = self.env
        obs, _ = env.reset()
        steps_done = 0
        while steps_done < total_timesteps:
            batch = self._collect_rollout(obs)
            obs = batch.pop("_last_obs")
            self._update(batch)
            steps_done += self.rollout_steps
        return self

    def _collect_rollout(self, obs):
        torch = _torch()
        env = self.env
        O, A, LP, R, V, D = [], [], [], [], [], []
        for _ in range(self.rollout_steps):
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self.device).reshape(1, -1)
            with torch.no_grad():
                dist = self._dist(obs_t)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = self.critic(obs_t).squeeze(-1)
            act = self._clip_action(a.cpu().numpy().ravel())
            nobs, r, term, trunc, _ = env.step(act)
            O.append(np.asarray(obs, dtype=np.float32))
            A.append(a.cpu().numpy().ravel())
            LP.append(float(logp.item()))
            R.append(float(r))
            V.append(float(v.item()))
            D.append(bool(term or trunc))
            obs = nobs
            if term or trunc:
                obs, _ = env.reset()
        # bootstrap value of the final obs
        with torch.no_grad():
            last_v = float(self.critic(
                torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device).reshape(1, -1)).item())
        adv, ret = self._gae(np.array(R), np.array(V), np.array(D), last_v)
        return {
            "obs": np.array(O, dtype=np.float32),
            "act": np.array(A, dtype=np.float32),
            "logp": np.array(LP, dtype=np.float32),
            "adv": adv.astype(np.float32),
            "ret": ret.astype(np.float32),
            "_last_obs": obs,
        }

    def _gae(self, rewards, values, dones, last_v):
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_v = last_v if t == n - 1 else values[t + 1]
            nonterm = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_v * nonterm - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + values
        return adv, ret

    def _update(self, batch):
        torch = _torch()
        dev = self.device
        obs = torch.as_tensor(batch["obs"], device=dev)
        act = torch.as_tensor(batch["act"], device=dev)
        old_logp = torch.as_tensor(batch["logp"], device=dev)
        adv = torch.as_tensor(batch["adv"], device=dev)
        ret = torch.as_tensor(batch["ret"], device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

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
                v = self.critic(obs[mb_t]).squeeze(-1)
                v_loss = ((v - ret[mb_t]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # --- persistence -----------------------------------------------------
    def _state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std.detach().cpu().numpy(),
        }

    def _load_state_dict(self, sd):
        if not sd:
            return
        torch = _torch()
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        with torch.no_grad():
            self.log_std.copy_(torch.as_tensor(sd["log_std"], device=self.device))
