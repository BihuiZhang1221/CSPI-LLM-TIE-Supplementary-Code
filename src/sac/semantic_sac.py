"""Semantic-gated SAC policy and replay buffer for the deployed controller."""

from __future__ import annotations

from array import array
from pathlib import Path
from typing import Any, NamedTuple, Optional
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
import copy
import math
from collections.abc import Iterable
from typing import Any, Optional
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import polyak_update
from torch import nn


ACTIVE_ROUTE_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
ROUTE_INFO_KEY = "base_route_id"
class ReplaySamples(NamedTuple):

    observations: dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: dict[str, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    route_ids: th.Tensor
class FrameReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str | th.device = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        *,
        active_route_ids: tuple[int, ...] | list[int] = ACTIVE_ROUTE_IDS,
        route_info_key: str = ROUTE_INFO_KEY,
        frame_pool_slack_per_env: int = 8,
        strict_frame_continuity: bool = True,
        frame_shape: tuple[int, int] = (45, 80),
        frame_stack: int = 4,
        state_dim: int = 28,
    ) -> None:
        if optimize_memory_usage:
            raise ValueError("base uses frame-index deduplication, cannot enable SB3 optimize_memory_usage.")
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("replay buffer requires a Dict observation space.")
        if set(observation_space.spaces) != {"ray_depth", "state"}:
            raise ValueError("base observation must contain exactly ray_depth and state.")
        if observation_space["ray_depth"].shape != (4, 45, 80):
            raise ValueError("ray_depth shape must be [4,45,80].")
        if observation_space["state"].shape != (28,):
            raise ValueError("state shape must be [28].")
        if not isinstance(action_space, spaces.Box) or action_space.shape != (5,):
            raise ValueError("action space must be a 5D Box.")
        if int(buffer_size) <= 0 or int(n_envs) <= 0:
            raise ValueError("buffer_size and n_envs must be positive.")

        routes = tuple(int(route) for route in active_route_ids)
        if routes != ACTIVE_ROUTE_IDS:
            raise ValueError(f"active routes must be exactly {ACTIVE_ROUTE_IDS}, got {routes}.")
        if tuple(frame_shape) != (45, 80) or int(frame_stack) != 4 or int(state_dim) != 28:
            raise ValueError("base fixed dimensions: frame_shape=(45,80), frame_stack=4, state_dim=28.")

        BaseBuffer.__init__(
            self,
            buffer_size=int(buffer_size),
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=int(n_envs),
        )
        self.optimize_memory_usage = False
        self.handle_timeout_termination = False
        self.active_route_ids = routes
        self.route_info_key = str(route_info_key)
        self.strict_frame_continuity = bool(strict_frame_continuity)

        self.frame_shape = (45, 80)
        self.history_length = 4
        self.frame_pool_capacity = int(buffer_size) + int(frame_pool_slack_per_env) * int(n_envs) + 8

        self.frames = np.empty((self.frame_pool_capacity, *self.frame_shape), dtype=np.uint16)
        self.frame_ref_counts = np.zeros(self.frame_pool_capacity, dtype=np.int32)
        self.frame_episode_ids = np.full(self.frame_pool_capacity, -1, dtype=np.int64)
        self._free_frame_ids = array("i", range(self.frame_pool_capacity - 1, -1, -1))

        capacity = int(buffer_size)
        self.obs_frame_ids = np.full((capacity, self.history_length), -1, dtype=np.int32)
        self.next_frame_ids = np.full((capacity, self.history_length), -1, dtype=np.int32)
        self.states = np.empty((capacity, 28), dtype=np.float32)
        self.next_states = np.empty((capacity, 28), dtype=np.float32)
        self.actions = np.empty((capacity, 5), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.route_ids = np.full(capacity, -1, dtype=np.int8)
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self._valid = np.zeros(capacity, dtype=np.bool_)

        self._route_slots: dict[int, array] = {route: array("i") for route in routes}
        self._slot_bucket_positions = np.full(capacity, -1, dtype=np.int32)
        self._quota_cursor = 0

        self._active_stack_ids = np.full((n_envs, self.history_length), -1, dtype=np.int32)
        self._active_episode_ids = np.full(n_envs, -1, dtype=np.int64)
        self._next_episode_serial = 0
        for env_index in range(n_envs):
            self._active_episode_ids[env_index] = self._new_episode_id()

        self.total_transitions_added = 0

    def _new_episode_id(self) -> int:

        episode_id = int(self._next_episode_serial)
        self._next_episode_serial += 1
        return episode_id

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> np.ndarray:

        frame_array = np.asarray(frame, dtype=np.float32)
        if frame_array.shape != (45, 80):
            raise ValueError(f"single-frame shape should be (45,80), got {frame_array.shape}.")
        if not np.isfinite(frame_array).all():
            raise ValueError("ray_depth contains NaN or Inf; refusing to pollute the replay buffer.")
        if float(frame_array.min()) < -1.0e-5 or float(frame_array.max()) > 1.00001:
            raise ValueError("ray_depth must already be log1p-encoded to [0,1].")
        return np.rint(np.clip(frame_array, 0.0, 1.0) * 65535.0).astype(np.uint16)

    def _allocate_encoded_frame(self, encoded: np.ndarray, episode_id: int) -> int:

        if not self._free_frame_ids:
            raise RuntimeError(
                "frame pool exhausted; this should not happen for a normal four-frame window. Check whether windows break after done or a reference count is wrong."
            )
        frame_id = int(self._free_frame_ids.pop())
        if int(self.frame_ref_counts[frame_id]) != 0:
            raise RuntimeError("frame free list corrupted: evicted frame is still referenced.")
        self.frames[frame_id] = encoded
        self.frame_episode_ids[frame_id] = int(episode_id)
        return frame_id

    def _allocate_reset_stack(self, stack: np.ndarray, episode_id: int) -> np.ndarray:

        ids = np.empty(self.history_length, dtype=np.int32)
        local_unique: dict[bytes, int] = {}
        for history_index in range(self.history_length):
            encoded = self._encode_frame(stack[history_index])
            key = encoded.tobytes()
            frame_id = local_unique.get(key)
            if frame_id is None:
                frame_id = self._allocate_encoded_frame(encoded, episode_id)
                local_unique[key] = frame_id
            ids[history_index] = frame_id
        return ids

    def _change_frame_refs(self, ids: np.ndarray, direction: int) -> None:

        unique_ids, counts = np.unique(np.asarray(ids, dtype=np.int32), return_counts=True)
        for frame_id, count in zip(unique_ids.tolist(), counts.tolist()):
            if frame_id < 0:
                raise RuntimeError("transition contains an invalid frame index.")
            new_count = int(self.frame_ref_counts[frame_id]) + int(direction) * int(count)
            if new_count < 0:
                raise RuntimeError("frame reference count underflow.")
            self.frame_ref_counts[frame_id] = new_count
            if new_count == 0 and direction < 0:
                self.frame_episode_ids[frame_id] = -1
                self._free_frame_ids.append(int(frame_id))

    def _remove_route_slot(self, slot: int) -> None:

        route_id = int(self.route_ids[slot])
        bucket = self._route_slots[route_id]
        position = int(self._slot_bucket_positions[slot])
        if position < 0 or position >= len(bucket) or int(bucket[position]) != slot:
            raise RuntimeError("route-bucket reverse index corrupted.")
        last_slot = int(bucket[-1])
        bucket[position] = last_slot
        self._slot_bucket_positions[last_slot] = position
        bucket.pop()
        self._slot_bucket_positions[slot] = -1

    def _evict_slot(self, slot: int) -> None:

        if not bool(self._valid[slot]):
            return
        self._remove_route_slot(slot)
        self._change_frame_refs(self.obs_frame_ids[slot], -1)
        self._change_frame_refs(self.next_frame_ids[slot], -1)
        self._valid[slot] = False

    def _extract_route_id(self, info: dict[str, Any], done: bool) -> int:

        candidates: list[Any] = []
        if done:
            for terminal_key in ("base_terminal", "bsa_d2_terminal"):
                terminal_info = info.get(terminal_key)
                if isinstance(terminal_info, dict):
                    candidates.extend(
                        terminal_info.get(key)
                        for key in ("base_route_id", "route_id")
                        if terminal_info.get(key) is not None
                    )
        candidates.extend(
            info.get(key)
            for key in ("base_transition_route_id", self.route_info_key, "route_id")
            if info.get(key) is not None
        )
        if not candidates:
            raise KeyError(
                f"info missing {self.route_info_key!r}; base wrapper must attach a route id to every transition."
            )
        value = candidates[0]
        if hasattr(value, "item"):
            value = value.item()
        route_id = int(value)
        if route_id not in self._route_slots:
            raise ValueError(f"replay received a disabled route {route_id}; only {self.active_route_ids} are allowed.")
        return route_id

    def _assert_stack_episode(self, stack_ids: np.ndarray, episode_id: int) -> None:

        frame_episodes = self.frame_episode_ids[np.asarray(stack_ids, dtype=np.int32)]
        if not np.all(frame_episodes == int(episode_id)):
            raise RuntimeError(
                f"cross-episode depth history detected: expected {episode_id}, got {frame_episodes.tolist()}."
            )

    def _validate_current_stack(self, stack: np.ndarray, ids: np.ndarray) -> None:

        if not self.strict_frame_continuity:
            return
        for history_index, frame_id in enumerate(ids.tolist()):
            encoded = self._encode_frame(stack[history_index])
            if not np.array_equal(encoded, self.frames[frame_id]):
                raise RuntimeError(
                    "ray_depth time window is discontinuous; a reset without done may have occurred,"
                    f"env history index={history_index}."
                )

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:

        ray_obs = np.asarray(obs["ray_depth"], dtype=np.float32)
        ray_next = np.asarray(next_obs["ray_depth"], dtype=np.float32)
        state_obs = np.asarray(obs["state"], dtype=np.float32)
        state_next = np.asarray(next_obs["state"], dtype=np.float32)
        expected_ray_shape = (self.n_envs, 4, 45, 80)
        if ray_obs.shape != expected_ray_shape or ray_next.shape != expected_ray_shape:
            raise ValueError(f"vector ray_depth shape should be {expected_ray_shape}.")
        if state_obs.shape != (self.n_envs, 28) or state_next.shape != (self.n_envs, 28):
            raise ValueError("vector state shape should be [n_envs,28].")
        if not np.isfinite(state_obs).all() or not np.isfinite(state_next).all():
            raise ValueError("state contains NaN or Inf.")

        actions = np.asarray(action, dtype=np.float32).reshape(self.n_envs, self.action_dim)
        rewards = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        dones = np.asarray(done, dtype=np.bool_).reshape(self.n_envs)
        if len(infos) != self.n_envs:
            raise ValueError("infos length must equal n_envs.")

        for env_index in range(self.n_envs):
            slot = int(self.pos)
            self._evict_slot(slot)

            episode_id = int(self._active_episode_ids[env_index])
            active_ids = self._active_stack_ids[env_index]
            if int(active_ids[0]) < 0:
                active_ids = self._allocate_reset_stack(ray_obs[env_index], episode_id)
                self._active_stack_ids[env_index] = active_ids
            else:
                self._validate_current_stack(ray_obs[env_index], active_ids)
            self._assert_stack_episode(active_ids, episode_id)

            is_done = bool(dones[env_index])
            route_id = self._extract_route_id(infos[env_index], is_done)
            obs_ids = np.array(active_ids, dtype=np.int32, copy=True)

            if is_done:
                next_ids = np.array(obs_ids, copy=True)
            else:
                if self.strict_frame_continuity:
                    for history_index in range(3):
                        encoded = self._encode_frame(ray_next[env_index, history_index])
                        if not np.array_equal(encoded, self.frames[obs_ids[history_index + 1]]):
                            raise RuntimeError("the first three frames of next ray_depth are not the shifted current window.")
                encoded_new = self._encode_frame(ray_next[env_index, -1])
                new_frame_id = self._allocate_encoded_frame(encoded_new, episode_id)
                next_ids = np.concatenate((obs_ids[1:], np.asarray([new_frame_id], dtype=np.int32)))
                self._assert_stack_episode(next_ids, episode_id)

            self.obs_frame_ids[slot] = obs_ids
            self.next_frame_ids[slot] = next_ids
            self.states[slot] = state_obs[env_index]
            self.next_states[slot] = state_next[env_index]
            self.actions[slot] = actions[env_index]
            self.rewards[slot] = rewards[env_index]
            self.dones[slot] = float(is_done)
            self.route_ids[slot] = route_id
            self.episode_ids[slot] = episode_id
            self._change_frame_refs(obs_ids, +1)
            self._change_frame_refs(next_ids, +1)

            bucket = self._route_slots[route_id]
            self._slot_bucket_positions[slot] = len(bucket)
            bucket.append(slot)
            self._valid[slot] = True

            if is_done:
                self._active_stack_ids[env_index].fill(-1)
                self._active_episode_ids[env_index] = self._new_episode_id()
            else:
                self._active_stack_ids[env_index] = next_ids

            self.total_transitions_added += 1
            self.pos += 1
            if self.pos >= self.buffer_size:
                self.pos = 0
                self.full = True

    def _balanced_slots(self, batch_size: int) -> np.ndarray:

        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        route_count = len(self.active_route_ids)
        base, remainder = divmod(int(batch_size), route_count)
        quotas = {route: base for route in self.active_route_ids}
        for offset in range(remainder):
            route = self.active_route_ids[(self._quota_cursor + offset) % route_count]
            quotas[route] += 1
        self._quota_cursor = (self._quota_cursor + remainder) % route_count

        selected: list[int] = []
        for route in self.active_route_ids:
            bucket = self._route_slots[route]
            count = quotas[route]
            if count and not bucket:
                raise RuntimeError(f"route {route} replay bucket is empty; cannot run seven-route balanced training yet.")
            if count:
                random_positions = np.random.randint(0, len(bucket), size=count)
                selected.extend(int(bucket[int(position)]) for position in random_positions)
        slots = np.asarray(selected, dtype=np.int64)
        np.random.shuffle(slots)
        return slots

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplaySamples:

        if env is not None:
            raise ValueError("VecNormalize is forbidden; depth and state are already encoded at fixed physical scales.")
        slots = self._balanced_slots(batch_size)
        return self._get_samples(slots, env=None)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplaySamples:

        if env is not None:
            raise ValueError("replay does not support VecNormalize.")
        slots = np.asarray(batch_inds, dtype=np.int64)
        if slots.ndim != 1 or not np.all(self._valid[slots]):
            raise ValueError("sampling index contains an unwritten transition.")

        obs_depth = self.frames[self.obs_frame_ids[slots]].astype(np.float32) / 65535.0
        next_depth = self.frames[self.next_frame_ids[slots]].astype(np.float32) / 65535.0
        observations = {
            "ray_depth": self.to_torch(np.ascontiguousarray(obs_depth)),
            "state": self.to_torch(np.ascontiguousarray(self.states[slots])),
        }
        next_observations = {
            "ray_depth": self.to_torch(np.ascontiguousarray(next_depth)),
            "state": self.to_torch(np.ascontiguousarray(self.next_states[slots])),
        }
        return ReplaySamples(
            observations=observations,
            actions=self.to_torch(np.ascontiguousarray(self.actions[slots])),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[slots].reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[slots].reshape(-1, 1)),
            route_ids=self.to_torch(self.route_ids[slots].astype(np.int64).reshape(-1, 1)),
        )

    def ready_for_training(self) -> bool:

        return all(len(self._route_slots[route]) > 0 for route in self.active_route_ids)

    def prepare_for_env_reset(self) -> None:

        self._active_stack_ids.fill(-1)
        for env_index in range(self.n_envs):
            self._active_episode_ids[env_index] = self._new_episode_id()

    def reset(self) -> None:

        self.pos = 0
        self.full = False
        self._valid.fill(False)
        self.obs_frame_ids.fill(-1)
        self.next_frame_ids.fill(-1)
        self.route_ids.fill(-1)
        self.episode_ids.fill(-1)
        self.frame_ref_counts.fill(0)
        self.frame_episode_ids.fill(-1)
        self._free_frame_ids = array("i", range(self.frame_pool_capacity - 1, -1, -1))
        for bucket in self._route_slots.values():
            del bucket[:]
        self._slot_bucket_positions.fill(-1)
        self._quota_cursor = 0
        self._active_stack_ids.fill(-1)
        self._next_episode_serial = 0
        for env_index in range(self.n_envs):
            self._active_episode_ids[env_index] = self._new_episode_id()
        self.total_transitions_added = 0

    def diagnostics(self) -> dict[str, float]:

        result = {
            "transitions": float(self.size()),
            "total_added": float(self.total_transitions_added),
            "frames_in_use": float(np.count_nonzero(self.frame_ref_counts)),
            "frame_pool_capacity": float(self.frame_pool_capacity),
            "frame_storage_gib": float(self.frames.nbytes / (1024**3)),
        }
        for route in self.active_route_ids:
            result[f"route_{route}_count"] = float(len(self._route_slots[route]))
        return result

    def save_snapshot(self, path: str | Path) -> None:

        save_to_pkl(Path(path), self, verbose=0)

    @classmethod
    def load_snapshot(
        cls, path: str | Path, device: str | th.device = "auto"
    ) -> "FrameReplayBuffer":

        replay = load_from_pkl(Path(path), verbose=0)
        if not isinstance(replay, cls):
            raise TypeError(f"snapshot is not a {cls.__name__}: {type(replay).__name__}.")
        replay.device = get_device(device)
        replay.prepare_for_env_reset()
        return replay
STATE_DIM = 38
BASE_STATE_DIM = 28
SEMANTIC_DIM = 10
class SemanticFrameReplayBuffer(FrameReplayBuffer):

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str | th.device = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        *,
        active_route_ids: tuple[int, ...] | list[int] = ACTIVE_ROUTE_IDS,
        route_info_key: str = ROUTE_INFO_KEY,
        frame_pool_slack_per_env: int = 8,
        strict_frame_continuity: bool = True,
        frame_shape: tuple[int, int] = (45, 80),
        frame_stack: int = 4,
        state_dim: int = STATE_DIM,
    ) -> None:

        if optimize_memory_usage:
            raise ValueError("semantic uses frame-index deduplication, cannot enable SB3 optimize_memory_usage.")
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("replay buffer requires a Dict observation space.")
        if set(observation_space.spaces) != {"ray_depth", "state"}:
            raise ValueError("semantic observation must contain exactly ray_depth and state.")
        if observation_space["ray_depth"].shape != (4, 45, 80):
            raise ValueError("ray_depth shape must be [4,45,80].")
        if observation_space["state"].shape != (STATE_DIM,):
            raise ValueError("state shape must be [38].")
        if not isinstance(action_space, spaces.Box) or action_space.shape != (5,):
            raise ValueError("action space must be a 5D Box.")
        if int(buffer_size) <= 0 or int(n_envs) <= 0:
            raise ValueError("buffer_size and n_envs must be positive.")

        routes = tuple(int(route) for route in active_route_ids)
        if routes != ACTIVE_ROUTE_IDS:
            raise ValueError(f"active routes must be exactly {ACTIVE_ROUTE_IDS}, got {routes}.")
        if tuple(frame_shape) != (45, 80) or int(frame_stack) != 4 or int(state_dim) != STATE_DIM:
            raise ValueError("semantic fixed sizes are frame_shape=(45,80), frame_stack=4, state_dim=38.")

        BaseBuffer.__init__(
            self,
            buffer_size=int(buffer_size),
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=int(n_envs),
        )
        self.optimize_memory_usage = False
        self.handle_timeout_termination = False
        self.active_route_ids = routes
        self.route_info_key = str(route_info_key)
        self.strict_frame_continuity = bool(strict_frame_continuity)
        self.state_dim = STATE_DIM

        self.frame_shape = (45, 80)
        self.history_length = 4
        self.frame_pool_capacity = int(buffer_size) + int(frame_pool_slack_per_env) * int(n_envs) + 8

        self.frames = np.empty((self.frame_pool_capacity, *self.frame_shape), dtype=np.uint16)
        self.frame_ref_counts = np.zeros(self.frame_pool_capacity, dtype=np.int32)
        self.frame_episode_ids = np.full(self.frame_pool_capacity, -1, dtype=np.int64)
        self._free_frame_ids = array("i", range(self.frame_pool_capacity - 1, -1, -1))

        capacity = int(buffer_size)
        self.obs_frame_ids = np.full((capacity, self.history_length), -1, dtype=np.int32)
        self.next_frame_ids = np.full((capacity, self.history_length), -1, dtype=np.int32)
        self.states = np.empty((capacity, STATE_DIM), dtype=np.float32)
        self.next_states = np.empty((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.empty((capacity, 5), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.route_ids = np.full(capacity, -1, dtype=np.int8)
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self._valid = np.zeros(capacity, dtype=np.bool_)

        self._route_slots: dict[int, array] = {route: array("i") for route in routes}
        self._slot_bucket_positions = np.full(capacity, -1, dtype=np.int32)
        self._quota_cursor = 0

        self._active_stack_ids = np.full((n_envs, self.history_length), -1, dtype=np.int32)
        self._active_episode_ids = np.full(n_envs, -1, dtype=np.int64)
        self._next_episode_serial = 0
        for env_index in range(n_envs):
            self._active_episode_ids[env_index] = self._new_episode_id()

        self.total_transitions_added = 0

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:

        ray_obs = np.asarray(obs["ray_depth"], dtype=np.float32)
        ray_next = np.asarray(next_obs["ray_depth"], dtype=np.float32)
        state_obs = np.asarray(obs["state"], dtype=np.float32)
        state_next = np.asarray(next_obs["state"], dtype=np.float32)
        expected_ray_shape = (self.n_envs, 4, 45, 80)
        expected_state_shape = (self.n_envs, STATE_DIM)
        if ray_obs.shape != expected_ray_shape or ray_next.shape != expected_ray_shape:
            raise ValueError(f"vector ray_depth shape should be {expected_ray_shape}.")
        if state_obs.shape != expected_state_shape or state_next.shape != expected_state_shape:
            raise ValueError(f"vector state shape should be {expected_state_shape}.")
        if not np.isfinite(state_obs).all() or not np.isfinite(state_next).all():
            raise ValueError("semantic state contains NaN or Inf.")

        actions = np.asarray(action, dtype=np.float32).reshape(self.n_envs, self.action_dim)
        rewards = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        dones = np.asarray(done, dtype=np.bool_).reshape(self.n_envs)
        if len(infos) != self.n_envs:
            raise ValueError("infos length must equal n_envs.")

        for env_index in range(self.n_envs):
            slot = int(self.pos)
            self._evict_slot(slot)

            episode_id = int(self._active_episode_ids[env_index])
            active_ids = self._active_stack_ids[env_index]
            if int(active_ids[0]) < 0:
                active_ids = self._allocate_reset_stack(ray_obs[env_index], episode_id)
                self._active_stack_ids[env_index] = active_ids
            else:
                self._validate_current_stack(ray_obs[env_index], active_ids)
            self._assert_stack_episode(active_ids, episode_id)

            is_done = bool(dones[env_index])
            route_id = self._extract_route_id(infos[env_index], is_done)
            obs_ids = np.array(active_ids, dtype=np.int32, copy=True)

            if is_done:
                next_ids = np.array(obs_ids, copy=True)
            else:
                if self.strict_frame_continuity:
                    for history_index in range(3):
                        encoded = self._encode_frame(ray_next[env_index, history_index])
                        if not np.array_equal(encoded, self.frames[obs_ids[history_index + 1]]):
                            raise RuntimeError("the first three frames of next ray_depth are not the shifted current window.")
                encoded_new = self._encode_frame(ray_next[env_index, -1])
                new_frame_id = self._allocate_encoded_frame(encoded_new, episode_id)
                next_ids = np.concatenate((obs_ids[1:], np.asarray([new_frame_id], dtype=np.int32)))
                self._assert_stack_episode(next_ids, episode_id)

            self.obs_frame_ids[slot] = obs_ids
            self.next_frame_ids[slot] = next_ids
            self.states[slot] = state_obs[env_index]
            self.next_states[slot] = state_next[env_index]
            self.actions[slot] = actions[env_index]
            self.rewards[slot] = rewards[env_index]
            self.dones[slot] = float(is_done)
            self.route_ids[slot] = route_id
            self.episode_ids[slot] = episode_id
            self._change_frame_refs(obs_ids, +1)
            self._change_frame_refs(next_ids, +1)

            bucket = self._route_slots[route_id]
            self._slot_bucket_positions[slot] = len(bucket)
            bucket.append(slot)
            self._valid[slot] = True

            if is_done:
                self._active_stack_ids[env_index].fill(-1)
                self._active_episode_ids[env_index] = self._new_episode_id()
            else:
                self._active_stack_ids[env_index] = next_ids

            self.total_transitions_added += 1
            self.pos += 1
            if self.pos >= self.buffer_size:
                self.pos = 0
                self.full = True

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplaySamples:

        return super()._get_samples(batch_inds, env=env)


AUV_LOG_STD_MIN = -5.0
AUV_LOG_STD_MAX = 1.0
AUV_ALPHA_MIN = 1.0e-4
AUV_ALPHA_MAX = 0.20
def _orthogonal_hidden_auv(module: nn.Module) -> None:

    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
FUSED_DIM = 256
NAV_DIM = 64
def _validate_semantic_obs_space(observation_space: spaces.Space) -> None:

    if not isinstance(observation_space, spaces.Dict):
        raise TypeError("semantic policy requires a Dict observation space.")
    if set(observation_space.spaces) != {"ray_depth", "state"}:
        raise ValueError("semantic observation must contain exactly ray_depth and state.")
    if observation_space["ray_depth"].shape != (4, 45, 80):
        raise ValueError("ray_depth shape must be [4,45,80].")
    if observation_space["state"].shape != (STATE_DIM,):
        raise ValueError("semantic state shape must be (38,).")
def _split_state(state: th.Tensor) -> tuple[th.Tensor, th.Tensor]:

    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"semantic state input must be [B,38], got {tuple(state.shape)}.")
    base_state = state[:, :BASE_STATE_DIM].float()
    semantic = state[:, BASE_STATE_DIM : BASE_STATE_DIM + SEMANTIC_DIM].float()
    return base_state, semantic
class VisualStream(nn.Module):

    output_dim = 128

    def __init__(
        self,
        in_channels: int = 1,
        frames: int = 4,
        patch: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.frames = int(frames)
        self.patch = int(patch)
        self.d_model = int(d_model)
        self.patch_dim = self.in_channels * self.patch * self.patch
        self.projection = nn.Linear(self.patch_dim, self.d_model)
        self.frame_embedding = nn.Parameter(th.zeros(1, self.frames, 1, self.d_model))
        self.position_embedding = nn.Parameter(th.zeros(1, 1, 256, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=2 * self.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        _orthogonal_hidden_auv(self)

    def forward(self, ray_depth: th.Tensor) -> th.Tensor:
        values = ray_depth.float()
        if values.ndim == 4:
            values = values.unsqueeze(2)
        if (
            values.ndim != 5
            or int(values.shape[1]) != self.frames
            or int(values.shape[2]) != self.in_channels
        ):
            raise ValueError(
                f"visual input must be [B,{self.frames},{self.in_channels},H,W], got {tuple(values.shape)}."
            )
        batch = values.shape[0]
        height, width = int(values.shape[3]), int(values.shape[4])
        pad_h = (self.patch - height % self.patch) % self.patch
        pad_w = (self.patch - width % self.patch) % self.patch
        if pad_h or pad_w:
            values = F.pad(values, (0, pad_w, 0, pad_h))
        patches = values.unfold(3, self.patch, self.patch).unfold(4, self.patch, self.patch)
        grid_h, grid_w = int(patches.shape[3]), int(patches.shape[4])
        patches = patches.permute(0, 1, 3, 4, 2, 5, 6).reshape(
            batch, self.frames, grid_h * grid_w, self.patch_dim
        )
        tokens = self.projection(patches)
        tokens = tokens + self.frame_embedding + self.position_embedding[:, :, : grid_h * grid_w]
        sequence = tokens.reshape(batch, self.frames * grid_h * grid_w, self.d_model)
        encoded = self.encoder(sequence)
        return self.norm(encoded.mean(dim=1))
class NavigationEncoder(nn.Module):

    output_dim = NAV_DIM

    _GROUPS = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 13), (13, 19), (19, 22), (22, 23), (23, 28))

    def __init__(
        self,
        base_state_dim: int = BASE_STATE_DIM,
        d_model: int = NAV_DIM,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.base_state_dim = int(base_state_dim)
        self.d_model = int(d_model)
        self.projections = nn.ModuleList(
            [nn.Linear(end - start, self.d_model) for start, end in self._GROUPS]
        )
        self.cls_token = nn.Parameter(th.zeros(1, 1, self.d_model))
        self.type_embedding = nn.Parameter(th.zeros(1, len(self._GROUPS) + 1, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=2 * self.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        _orthogonal_hidden_auv(self)

    def forward(self, base_state: th.Tensor) -> th.Tensor:
        if base_state.ndim != 2 or base_state.shape[1] != self.base_state_dim:
            raise ValueError(
                f"base_state input must be [B,{self.base_state_dim}], got {tuple(base_state.shape)}."
            )
        state = base_state.float()
        batch = state.shape[0]
        tokens = [self.cls_token.expand(batch, -1, -1)]
        for (start, end), projection in zip(self._GROUPS, self.projections):
            tokens.append(projection(state[:, start:end]).unsqueeze(1))
        sequence = th.cat(tokens, dim=1) + self.type_embedding
        encoded = self.encoder(sequence)
        return self.norm(encoded[:, 0])
class DualStreamEncoder(nn.Module):

    def __init__(
        self,
        visual_encoder: nn.Module,
        *,
        use_dual_stream: bool = True,
        base_state_dim: int = BASE_STATE_DIM,
        nav_dim: int = NAV_DIM,
        fused_dim: int = FUSED_DIM,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_dual_stream = bool(use_dual_stream)
        self.base_state_dim = int(base_state_dim)
        self.nav_dim = int(nav_dim)
        self.fused_dim = int(fused_dim)
        self.visual_dim = int(getattr(visual_encoder, "output_dim", 128))
        self.navigation_encoder = (
            NavigationEncoder(self.base_state_dim, self.nav_dim) if self.use_dual_stream else None
        )
        self.nav_projection = (
            nn.Linear(self.nav_dim, self.fused_dim) if self.use_dual_stream else None
        )
        self.state_projection = (
            None if self.use_dual_stream else nn.Linear(self.base_state_dim, self.fused_dim)
        )
        self.visual_projection = nn.Linear(self.visual_dim, self.fused_dim)
        self.cls_token = nn.Parameter(th.zeros(1, 1, self.fused_dim))
        self.type_embedding = nn.Parameter(th.zeros(1, 3, self.fused_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.fused_dim,
            nhead=int(nhead),
            dim_feedforward=2 * self.fused_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.fused_dim)
        _orthogonal_hidden_auv(self)

    def non_visual_parameters(self) -> Iterable[nn.Parameter]:
        if self.navigation_encoder is not None:
            yield from self.navigation_encoder.parameters()
        if self.nav_projection is not None:
            yield from self.nav_projection.parameters()
        if self.state_projection is not None:
            yield from self.state_projection.parameters()
        yield from self.visual_projection.parameters()
        yield self.cls_token
        yield self.type_embedding
        yield from self.fusion.parameters()
        yield from self.norm.parameters()

    def fuse_from_features(self, visual_features: th.Tensor, base_state: th.Tensor) -> th.Tensor:
        if visual_features.ndim != 2 or visual_features.shape[1] != self.visual_dim:
            raise ValueError(
                f"visual feature must be [B,{self.visual_dim}], got {tuple(visual_features.shape)}."
            )
        if base_state.ndim != 2 or base_state.shape[1] != self.base_state_dim:
            raise ValueError(
                f"base_state must be [B,{self.base_state_dim}], got {tuple(base_state.shape)}."
            )
        batch = base_state.shape[0]
        visual_token = self.visual_projection(visual_features.float())
        if self.use_dual_stream:
            nav_features = self.navigation_encoder(base_state)
            nav_token = self.nav_projection(nav_features)
        else:
            nav_token = self.state_projection(base_state.float())
        tokens = th.cat(
            [self.cls_token.expand(batch, -1, -1), visual_token.unsqueeze(1), nav_token.unsqueeze(1)],
            dim=1,
        ) + self.type_embedding
        fused = self.fusion(tokens)
        return self.norm(fused[:, 0])
class SemanticGate(nn.Module):

    def __init__(
        self,
        semantic_dim: int = SEMANTIC_DIM,
        latent_dim: int = FUSED_DIM,
        hidden_dim: int = 64,
        gate_scale: float = 0.25,
        use_semantic_gate: bool = True,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.latent_dim = int(latent_dim)
        self.gate_scale = float(gate_scale)
        self.use_semantic_gate = bool(use_semantic_gate)
        self.network = nn.Sequential(
            nn.Linear(self.semantic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.Tanh(),
        )
        _orthogonal_hidden_auv(self)
        final_linear = self.network[-2]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, latent: th.Tensor, semantic: th.Tensor) -> th.Tensor:
        if not self.use_semantic_gate:
            return latent
        if semantic.ndim != 2 or semantic.shape[1] != self.semantic_dim:
            raise ValueError(f"semantic input must be [B,{self.semantic_dim}], got {tuple(semantic.shape)}.")
        gate = self.network(semantic.float())
        return latent * (1.0 + self.gate_scale * gate)
class SemanticActor(nn.Module):

    def __init__(
        self,
        visual_encoder: VisualStream,
        action_dim: int = 5,
        *,
        use_dual_stream: bool = True,
        use_semantic_gate: bool = True,
        gate_scale: float = 0.25,
        log_std_min: float = AUV_LOG_STD_MIN,
        log_std_max: float = AUV_LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.dual_stream = DualStreamEncoder(visual_encoder, use_dual_stream=use_dual_stream)
        self.semantic_gate = SemanticGate(
            use_semantic_gate=use_semantic_gate,
            gate_scale=gate_scale,
        )
        self.trunk = nn.Sequential(
            nn.Linear(FUSED_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

        _orthogonal_hidden_auv(self.trunk)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, -0.5)

    def actor_parameters(self) -> Iterable[nn.Parameter]:

        yield from self.dual_stream.non_visual_parameters()
        yield from self.semantic_gate.parameters()
        yield from self.trunk.parameters()
        yield from self.mean.parameters()
        yield from self.log_std.parameters()

    def visual_features(self, observations: dict[str, th.Tensor]) -> th.Tensor:

        return self.visual_encoder(observations["ray_depth"]).detach()

    def latent(self, observations: dict[str, th.Tensor], *, visual_features: Optional[th.Tensor] = None) -> th.Tensor:
        base_state, semantic = _split_state(observations["state"])
        visual = self.visual_features(observations) if visual_features is None else visual_features
        fused = self.dual_stream.fuse_from_features(visual, base_state)
        return self.semantic_gate(fused, semantic)

    def distribution_parameters(self, observations: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        latent = self.trunk(self.latent(observations))
        mean = self.mean(latent)
        log_std = th.clamp(self.log_std(latent), self.log_std_min, self.log_std_max)
        return mean, log_std

    def forward(self, observations: dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        mean, log_std = self.distribution_parameters(observations)
        return self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic)

    def action_log_prob(self, observations: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        mean, log_std = self.distribution_parameters(observations)
        return self.action_dist.log_prob_from_params(mean, log_std)
class SemanticQNetwork(nn.Module):

    def __init__(self, latent_dim: int = FUSED_DIM, action_dim: int = 5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        _orthogonal_hidden_auv(self)
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.orthogonal_(final.weight, gain=1.0)
        nn.init.zeros_(final.bias)

    def forward(self, features_and_action: th.Tensor) -> th.Tensor:
        return self.network(features_and_action)
class SevenRouteCritic(nn.Module):

    def __init__(
        self,
        visual_encoder: VisualStream,
        active_route_ids: tuple[int, ...] = ACTIVE_ROUTE_IDS,
        *,
        use_dual_stream: bool = True,
        use_semantic_gate: bool = True,
        gate_scale: float = 0.25,
    ) -> None:
        super().__init__()
        routes = tuple(int(route) for route in active_route_ids)
        if routes != ACTIVE_ROUTE_IDS:
            raise ValueError(f"critic routes must be {ACTIVE_ROUTE_IDS}, got {routes}.")
        self.active_route_ids = routes
        self.visual_encoder = visual_encoder
        self.dual_stream = DualStreamEncoder(visual_encoder, use_dual_stream=use_dual_stream)
        self.semantic_gate = SemanticGate(
            use_semantic_gate=use_semantic_gate,
            gate_scale=gate_scale,
        )
        self.q1_heads = nn.ModuleDict({str(route): SemanticQNetwork() for route in routes})
        self.q2_heads = nn.ModuleDict({str(route): SemanticQNetwork() for route in routes})

        route_lookup = th.full((max(routes) + 1,), -1, dtype=th.long)
        for head_index, route in enumerate(routes):
            route_lookup[route] = head_index
        self.register_buffer("route_lookup", route_lookup, persistent=True)

    def _validate_routes(self, route_ids: th.Tensor, batch_size: int) -> th.Tensor:
        routes = route_ids.reshape(-1).long()
        if routes.numel() != batch_size:
            raise ValueError("route_ids count must equal the batch size.")
        in_range = (routes >= 0) & (routes < self.route_lookup.numel())
        valid = in_range.clone()
        valid[in_range] = self.route_lookup[routes[in_range]] >= 0
        if not bool(valid.all()):
            invalid = th.unique(routes[~valid]).detach().cpu().tolist()
            raise ValueError(f"critic received a disabled route {invalid}.")
        return routes

    def latent(
        self,
        observations: dict[str, th.Tensor],
        *,
        visual_features: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        base_state, semantic = _split_state(observations["state"])
        visual = self.visual_encoder(observations["ray_depth"]) if visual_features is None else visual_features
        fused = self.dual_stream.fuse_from_features(visual, base_state)
        return self.semantic_gate(fused, semantic)

    def forward(
        self,
        observations: dict[str, th.Tensor],
        actions: th.Tensor,
        route_ids: th.Tensor,
        *,
        visual_features: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        if actions.ndim != 2 or actions.shape[1] != 5:
            raise ValueError(f"critic action should be [B,5], got {tuple(actions.shape)}.")
        routes = self._validate_routes(route_ids, actions.shape[0])
        latent = self.latent(observations, visual_features=visual_features)
        inputs = th.cat((latent, actions.float()), dim=1)

        q1 = th.empty((actions.shape[0], 1), device=actions.device, dtype=inputs.dtype)
        q2 = th.empty_like(q1)
        for route in self.active_route_ids:
            indices = th.nonzero(routes == route, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            selected = inputs.index_select(0, indices)
            q1 = q1.index_copy(0, indices, self.q1_heads[str(route)](selected))
            q2 = q2.index_copy(0, indices, self.q2_heads[str(route)](selected))
        return q1, q2
class SemanticSACPolicy(BasePolicy):

    actor: SemanticActor
    critic: SevenRouteCritic
    critic_target: SevenRouteCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        *,
        active_route_ids: tuple[int, ...] | list[int] = ACTIVE_ROUTE_IDS,
        ray_shape: tuple[int, int, int] = (4, 45, 80),
        state_dim: int = STATE_DIM,
        feature_dim: int = 128,
        use_dual_stream: bool = True,
        use_semantic_gate: bool = True,
        gate_scale: float = 0.25,
        log_std_min: float = AUV_LOG_STD_MIN,
        log_std_max: float = AUV_LOG_STD_MAX,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = False,
        use_sde: bool = False,
    ) -> None:
        _validate_semantic_obs_space(observation_space)
        if not isinstance(action_space, spaces.Box) or action_space.shape != (5,):
            raise ValueError("policy action space must be a 5D Box.")
        if tuple(ray_shape) != (4, 45, 80) or int(state_dim) != STATE_DIM or int(feature_dim) != 128:
            raise ValueError("semantic fixed ray_shape=(4,45,80), state_dim=38, feature_dim=128.")
        if float(log_std_min) != -5.0 or float(log_std_max) != 1.0:
            raise ValueError("log_std range is fixed to [-5,1].")
        routes = tuple(int(route) for route in active_route_ids)
        if routes != ACTIVE_ROUTE_IDS:
            raise ValueError(f"policy routes must be {ACTIVE_ROUTE_IDS}.")
        if normalize_images:
            raise ValueError("ray_depth is already a [0,1] physical encoding; SB3 image division by 255 is forbidden.")
        if use_sde:
            raise ValueError("semantic uses standard tanh Gaussian SAC without gSDE.")

        optimizer_options = {"eps": 1.0e-5}
        if optimizer_kwargs:
            optimizer_options.update(optimizer_kwargs)
        super().__init__(
            observation_space,
            action_space,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_options,
            squash_output=True,
        )
        self.active_route_ids = routes
        self.ray_shape = tuple(ray_shape)
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.use_dual_stream = bool(use_dual_stream)
        self.use_semantic_gate = bool(use_semantic_gate)
        self.gate_scale = float(gate_scale)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.lr_schedule = lr_schedule

        shared_encoder = VisualStream()
        self.actor = SemanticActor(
            shared_encoder,
            use_dual_stream=self.use_dual_stream,
            use_semantic_gate=self.use_semantic_gate,
            gate_scale=self.gate_scale,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.critic = SevenRouteCritic(
            shared_encoder,
            routes,
            use_dual_stream=self.use_dual_stream,
            use_semantic_gate=self.use_semantic_gate,
            gate_scale=self.gate_scale,
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.critic_target.eval()

        learning_rate = float(lr_schedule(1.0))
        self.actor.optimizer = optimizer_class(
            list(self.actor.actor_parameters()), lr=learning_rate, **optimizer_options
        )
        self.critic.optimizer = optimizer_class(
            self.critic.parameters(), lr=learning_rate, **optimizer_options
        )
        self._assert_optimizer_ownership()

    def _assert_optimizer_ownership(self) -> None:

        cnn_ids = {id(parameter) for parameter in self.critic.visual_encoder.parameters()}
        actor_ids = {
            id(parameter)
            for group in self.actor.optimizer.param_groups
            for parameter in group["params"]
        }
        critic_ids = {
            id(parameter)
            for group in self.critic.optimizer.param_groups
            for parameter in group["params"]
        }
        if cnn_ids & actor_ids:
            raise RuntimeError("actor optimizer incorrectly includes shared CNN parameters.")
        if not cnn_ids <= critic_ids:
            raise RuntimeError("critic optimizer does not fully include shared CNN parameters.")

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        if not isinstance(observation, dict):
            raise TypeError("semantic policy prediction requires a dict observation.")
        return self.actor(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.eval()
        self.training = mode

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            lr_schedule=self._dummy_schedule,
            active_route_ids=self.active_route_ids,
            ray_shape=self.ray_shape,
            state_dim=self.state_dim,
            feature_dim=self.feature_dim,
            use_dual_stream=self.use_dual_stream,
            use_semantic_gate=self.use_semantic_gate,
            gate_scale=self.gate_scale,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            normalize_images=False,
        )
        return data
def _equal_route_mean_semantic(values: th.Tensor, route_ids: th.Tensor) -> th.Tensor:

    routes = route_ids.reshape(-1)
    means = []
    for route in ACTIVE_ROUTE_IDS:
        mask = routes == route
        if not bool(mask.any()):
            raise RuntimeError(f"training batch is missing route {route}; the replay balance constraint is broken.")
        means.append(values.reshape(-1)[mask].mean())
    return th.stack(means).mean()
class SemanticRouteCriticSAC(SAC):

    policy: SemanticSACPolicy
    actor: SemanticActor
    critic: SevenRouteCritic
    critic_target: SevenRouteCritic

    def __init__(
        self,
        policy: type[SemanticSACPolicy] | str = SemanticSACPolicy,
        env=None,
        learning_rate: float | Schedule = 3.0e-4,
        buffer_size: int = 300_000,
        learning_starts: int = 50_000,
        batch_size: int = 512,
        tau: float = 0.005,
        gamma: float = 0.995,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: type[SemanticFrameReplayBuffer] = SemanticFrameReplayBuffer,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        ent_coef: str | float = "auto_0.05",
        target_update_interval: int = 1,
        target_entropy: str | float = -5.0,
        policy_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(policy, str):
            if policy not in ("semanticPolicy", "SemanticSACPolicy"):
                raise ValueError(f"unsupported policy name {policy!r}.")
            policy = SemanticSACPolicy
        replay_options = dict(replay_buffer_kwargs or {})
        replay_options.setdefault("active_route_ids", ACTIVE_ROUTE_IDS)
        replay_options.setdefault("route_info_key", "base_route_id")
        replay_options.setdefault("handle_timeout_termination", False)
        replay_options["state_dim"] = STATE_DIM
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_options,
            optimize_memory_usage=False,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.policy, SemanticSACPolicy):
            raise TypeError("SemanticRouteCriticSAC must use SemanticSACPolicy.")
        if not isinstance(self.replay_buffer, SemanticFrameReplayBuffer):
            raise TypeError("SemanticRouteCriticSAC must use SemanticFrameReplayBuffer.")
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=float(self.lr_schedule(1.0)), eps=1.0e-5
            )

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:

        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def train(self, gradient_steps: int, batch_size: int = 512) -> None:

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        alpha_values: list[float] = []
        alpha_losses: list[float] = []
        critic_losses: list[float] = []
        actor_losses: list[float] = []

        assert isinstance(self.replay_buffer, SemanticFrameReplayBuffer)
        for _ in range(int(gradient_steps)):
            replay_data: ReplaySamples = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            _, alpha_log_prob = self.actor.action_log_prob(replay_data.observations)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                alpha = self.log_ent_coef.detach().exp()
                alpha_loss = -(
                    self.log_ent_coef * (alpha_log_prob + float(self.target_entropy)).detach()
                ).mean()
                self.ent_coef_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.ent_coef_optimizer.step()
                with th.no_grad():
                    self.log_ent_coef.clamp_(math.log(AUV_ALPHA_MIN), math.log(AUV_ALPHA_MAX))
                alpha_losses.append(float(alpha_loss.detach().cpu()))
                alpha = self.log_ent_coef.detach().exp()
            else:
                alpha = self.ent_coef_tensor
                if not AUV_ALPHA_MIN <= float(alpha) <= AUV_ALPHA_MAX:
                    raise ValueError(
                        f"fixed alpha must be within [{AUV_ALPHA_MIN:.4g},{AUV_ALPHA_MAX:.2f}]."
                    )
            alpha_values.append(float(alpha.cpu()))

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                target_q1, target_q2 = self.critic_target(
                    replay_data.next_observations, next_actions, replay_data.route_ids
                )
                target_min_q = th.minimum(target_q1, target_q2) - alpha * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * target_min_q

            current_q1, current_q2 = self.critic(
                replay_data.observations, replay_data.actions, replay_data.route_ids
            )
            per_sample_critic = 0.5 * (
                F.mse_loss(current_q1, target_q, reduction="none")
                + F.mse_loss(current_q2, target_q, reduction="none")
            )
            critic_loss = _equal_route_mean_semantic(per_sample_critic, replay_data.route_ids)
            self.critic.optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            self._set_requires_grad(self.critic, False)
            detached_visual = self.actor.visual_features(replay_data.observations)
            q1_pi, q2_pi = self.critic(
                replay_data.observations,
                actions_pi,
                replay_data.route_ids,
                visual_features=detached_visual,
            )
            actor_per_sample = alpha * log_prob.reshape(-1, 1) - th.minimum(q1_pi, q2_pi)
            actor_loss = _equal_route_mean_semantic(actor_per_sample, replay_data.route_ids)
            self.actor.optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor.optimizer.step()
            self._set_requires_grad(self.critic, True)
            self.critic_target.requires_grad_(False)
            actor_losses.append(float(actor_loss.detach().cpu()))

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(alpha_values)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        if alpha_losses:
            self.logger.record("train/ent_coef_loss", float(np.mean(alpha_losses)))
        for key, value in self.replay_buffer.diagnostics().items():
            self.logger.record(f"replay/{key}", value)
