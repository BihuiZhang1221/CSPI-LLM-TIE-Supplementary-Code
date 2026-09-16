"""Semantic observation wrappers for the inspection tasks."""

from __future__ import annotations

import contextlib
import csv
import datetime as dt
import io
import os
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from src.isaacsim.environment import AuvSb3VecEnvWrapper
from src.sac.semantic_sac import BASE_STATE_DIM, SEMANTIC_DIM, STATE_DIM
from src.teacher.teacher import (
    rule_semantic_from_observation,
    rule_semantic_from_observation_base,
)
from src.teacher.teacher import (
    FrameSnapshot,
    QwenJsonSemanticBatcher,
    QwenTextSemanticBatcher,
    build_json_semantic_prompt,
    build_text_semantic_prompt,
    capture_frame_snapshot,
    frame_debug_fields,
    parse_semantic_json_text,
    parse_semantic_text,
    semantic_text,
    semantic_tokens_to_vector,
    semantic_tokens_to_vector_base,
    semantic_vector_to_matrix_text,
    semantic_vector_to_matrix_text_base,
    semantic_vector_to_tokens,
)


DEFAULT_QWEN_BASE_TEXT = os.environ.get(
    "CSPI_QWEN_BASE",
    str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-4B"),
)
DEFAULT_QWEN_LORA_TEXT = os.environ.get(
    "CSPI_QWEN_LORA_TEXT",
    str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-4B-SFT-LoRA"),
)

EPISODE_LLM_LOG_FIELDS = (
    "episode_id",
    "timestamp",
    "transitions",
    "env_id",
    "route_id",
    "reason",
    "result",
    "episode_return",
    "llm_time_s",
    "llm_calls",
    "llm_raw",
    "llm_labels",
    "llm_matrix",
    "parse_failures",
)


@dataclass
class SemanticWrapperConfig:

    qwen_enable_after_transitions: int = 50_000
    qwen_period_steps: int = 5
    control_dt_s: float = 0.1
    qwen_batch_size: int = 8
    qwen_base_model_path: str = DEFAULT_QWEN_BASE_TEXT
    qwen_lora_adapter_path: str = DEFAULT_QWEN_LORA_TEXT
    qwen_device: str = "cuda"
    qwen_torch_dtype: str = "bfloat16"
    enable_qwen: bool = True
    mask_privileged_current: bool = False


class SemanticVecEnvWrapper(AuvSb3VecEnvWrapper):

    def __init__(
        self,
        env,
        episode_log_path: Path,
        *,
        fast_variant: bool = True,
        semantic_config: SemanticWrapperConfig | None = None,
    ) -> None:
        self.semantic_config = semantic_config or SemanticWrapperConfig()
        super().__init__(env, episode_log_path=episode_log_path, fast_variant=fast_variant)
        self._episode_llm_log_path = Path(episode_log_path).with_name("textsemantic_episode_llm.csv").resolve()
        self._ensure_episode_llm_log_header()
        self._semantic_control_step = 0
        self._last_actions = np.zeros((self.num_envs, 5), dtype=np.float32)
        self._semantic_vectors = np.zeros((self.num_envs, 9), dtype=np.float32)
        self._semantic_age = np.zeros(self.num_envs, dtype=np.float32)
        self._semantic_texts = np.full(self.num_envs, "CRUISE|ALIGN_GOAL", dtype=object)
        self._semantic_sources = np.full(self.num_envs, "uninitialized", dtype=object)
        self._semantic_parse_failed = np.zeros(self.num_envs, dtype=np.bool_)
        self._qwen_batcher: QwenTextSemanticBatcher | None = None
        self._last_qwen_metrics: dict[str, object] = {}
        self._episode_llm_time_ms = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_llm_calls = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_llm_parse_failures = np.zeros(self.num_envs, dtype=np.int64)
        self._last_llm_latency_ms = np.full(self.num_envs, np.nan, dtype=np.float64)
        self._last_llm_raw_texts = np.full(self.num_envs, None, dtype=object)
        self._last_llm_parsed_texts = np.full(self.num_envs, None, dtype=object)
        self._last_llm_matrix_texts = np.full(self.num_envs, None, dtype=object)

    def _reset_episode_llm_stats(self, env_indices: np.ndarray | list[int] | None = None) -> None:

        if env_indices is None:
            indices = np.arange(self.num_envs, dtype=np.int64)
        else:
            indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
        self._episode_llm_time_ms[indices] = 0.0
        self._episode_llm_calls[indices] = 0
        self._episode_llm_parse_failures[indices] = 0
        self._last_llm_latency_ms[indices] = np.nan
        self._last_llm_raw_texts[indices] = None
        self._last_llm_parsed_texts[indices] = None
        self._last_llm_matrix_texts[indices] = None

    def _ensure_episode_llm_log_header(self) -> None:

        self._episode_llm_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._episode_llm_log_path.is_file() and self._episode_llm_log_path.stat().st_size > 0:
            with self._episode_llm_log_path.open("r", encoding="utf-8", newline="") as stream:
                actual = tuple(next(csv.reader(stream), ()))
            if actual != EPISODE_LLM_LOG_FIELDS:
                raise RuntimeError(
                    "LLM episode CSV header mismatch:"
                    f"actual={actual}, expected={EPISODE_LLM_LOG_FIELDS}"
                )
            return

        with self._episode_llm_log_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=EPISODE_LLM_LOG_FIELDS).writeheader()
            stream.flush()

    def _append_episode_llm_log(self, record: dict[str, object]) -> None:

        missing = set(EPISODE_LLM_LOG_FIELDS) - set(record)
        extra = set(record) - set(EPISODE_LLM_LOG_FIELDS)
        if missing or extra:
            raise ValueError(f"semantic LLM CSV field error: missing={sorted(missing)} extra={sorted(extra)}")
        with self._episode_llm_log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=EPISODE_LLM_LOG_FIELDS).writerow(
                {field: record[field] for field in EPISODE_LLM_LOG_FIELDS}
            )
            stream.flush()

    def _process_spaces(self) -> None:

        observation_space = self.unwrapped.single_observation_space["policy"]
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("semantic must extend a Dict observation.")
        if set(observation_space.spaces) != {"ray_depth", "state"}:
            raise ValueError("semantic underlying observation may only contain ray_depth and state.")
        if observation_space["ray_depth"].shape != (4, 45, 80):
            raise ValueError("underlying ray_depth must be [4,45,80].")
        base_state_space = observation_space["state"]
        if not isinstance(base_state_space, spaces.Box) or base_state_space.shape != (BASE_STATE_DIM,):
            raise ValueError("underlying state must be a 28D Box.")

        semantic_low = np.zeros(SEMANTIC_DIM, dtype=np.float32)
        semantic_high = np.ones(SEMANTIC_DIM, dtype=np.float32)
        semantic_high[-1] = 10.0
        state_low = np.concatenate((base_state_space.low.astype(np.float32), semantic_low))
        state_high = np.concatenate((base_state_space.high.astype(np.float32), semantic_high))
        expanded_observation_space = spaces.Dict(
            {
                "ray_depth": observation_space["ray_depth"],
                "state": spaces.Box(low=state_low, high=state_high, dtype=np.float32),
            }
        )
        action_space = self.unwrapped.single_action_space
        if not isinstance(action_space, spaces.Box) or action_space.shape != (5,):
            raise ValueError("action must be a Box of shape (5,).")
        VecEnv.__init__(self, self.num_envs, expanded_observation_space, action_space)

    def _qwen(self) -> QwenTextSemanticBatcher:

        if self._qwen_batcher is None:
            cfg = self.semantic_config
            self._qwen_batcher = QwenTextSemanticBatcher(
                cfg.qwen_base_model_path,
                cfg.qwen_lora_adapter_path,
                device=cfg.qwen_device,
                torch_dtype=cfg.qwen_torch_dtype,
                qwen_batch_size=cfg.qwen_batch_size,
            )
        return self._qwen_batcher

    def _base_state_for_policy(self, observation: dict[str, np.ndarray]) -> np.ndarray:

        base_state = np.asarray(observation["state"], dtype=np.float32).copy()
        if base_state.shape != (self.num_envs, BASE_STATE_DIM):
            raise ValueError(f"semantic underlying state should be [n_envs,28], got {base_state.shape}.")
        if not np.isfinite(base_state).all():
            raise ValueError("underlying 28D state contains NaN or Inf; refusing to write to SAC/replay.")
        return base_state

    def _rule_refresh(self, observation: dict[str, np.ndarray], source: str) -> None:

        base_state = self._base_state_for_policy(observation)
        ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
        for env_index in range(self.num_envs):
            tokens = rule_semantic_from_observation_base(
                base_state[env_index],
                ray_depth[env_index],
                self._last_actions[env_index],
            )
            self._semantic_vectors[env_index] = semantic_tokens_to_vector_base(tokens)
            self._semantic_texts[env_index] = semantic_text(tokens)
            self._semantic_sources[env_index] = source
            self._semantic_parse_failed[env_index] = False
            self._semantic_age[env_index] = 0.0

    def _qwen_refresh(self, observation: dict[str, np.ndarray]) -> None:

        base_state = self._base_state_for_policy(observation)
        ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
        prompts = [
            build_text_semantic_prompt(
                base_state[env_index],
                ray_depth[env_index],
                self._last_actions[env_index],
                float(self._semantic_age[env_index]),
            )
            for env_index in range(self.num_envs)
        ]
        raw_texts, metrics = self._qwen().predict_texts(prompts)
        self._last_qwen_metrics = metrics
        prompt_latencies = metrics.get("qwen_prompt_latency_ms", [])
        for env_index, raw_text in enumerate(raw_texts):
            if isinstance(prompt_latencies, list) and env_index < len(prompt_latencies):
                latency_ms = float(prompt_latencies[env_index])
            else:
                latency_ms = float("nan")
            self._episode_llm_calls[env_index] += 1
            if np.isfinite(latency_ms):
                self._episode_llm_time_ms[env_index] += latency_ms
                self._last_llm_latency_ms[env_index] = latency_ms
            self._last_llm_raw_texts[env_index] = str(raw_text)

            parsed = parse_semantic_text(raw_text)
            if parsed.valid:
                self._semantic_vectors[env_index] = parsed.vector
                parsed_text = semantic_text(parsed.tokens)
                self._semantic_texts[env_index] = parsed_text
                self._semantic_sources[env_index] = "qwen"
                self._semantic_parse_failed[env_index] = False
                self._semantic_age[env_index] = 0.0
                self._last_llm_parsed_texts[env_index] = parsed_text
                self._last_llm_matrix_texts[env_index] = semantic_vector_to_matrix_text_base(parsed.vector)
            else:
                self._semantic_sources[env_index] = "hold_last_due_to_invalid_qwen"
                self._semantic_parse_failed[env_index] = True
                self._episode_llm_parse_failures[env_index] += 1
                self._last_llm_parsed_texts[env_index] = None
                self._last_llm_matrix_texts[env_index] = None

    def _maybe_refresh_semantic(self, observation: dict[str, np.ndarray]) -> None:

        cfg = self.semantic_config
        should_refresh = self._semantic_control_step % int(cfg.qwen_period_steps) == 0
        if not should_refresh:
            return
        if (not cfg.enable_qwen) or int(self._console_transitions) < int(cfg.qwen_enable_after_transitions):
            self._rule_refresh(observation, source="rule_warmup")
        else:
            self._qwen_refresh(observation)

    def _augment_observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:

        base_state = self._base_state_for_policy(observation)
        semantic_state = np.concatenate(
            (self._semantic_vectors, self._semantic_age.reshape(-1, 1)),
            axis=1,
        ).astype(np.float32)
        expanded_state = np.concatenate((base_state, semantic_state), axis=1).astype(np.float32)
        if expanded_state.shape != (self.num_envs, STATE_DIM):
            raise RuntimeError(f"semantic expanded state shape mismatch: {expanded_state.shape}")
        return {
            "ray_depth": np.asarray(observation["ray_depth"], dtype=np.float32),
            "state": expanded_state,
        }

    def _augment_single_terminal_observation(
        self,
        terminal_observation: dict[str, np.ndarray],
        semantic_vector: np.ndarray,
        semantic_age_s: float,
    ) -> dict[str, np.ndarray]:

        if not isinstance(terminal_observation, dict):
            raise TypeError("semantic terminal_observation must be Dict observation.")
        if set(terminal_observation.keys()) != {"ray_depth", "state"}:
            raise ValueError("semantic terminal_observation may only contain ray_depth and state.")

        terminal_state = np.asarray(terminal_observation["state"], dtype=np.float32).copy()
        if terminal_state.shape == (STATE_DIM,):
            expanded_state = terminal_state.astype(np.float32)
        elif terminal_state.shape == (BASE_STATE_DIM,):
            semantic_state = np.concatenate(
                (
                    np.asarray(semantic_vector, dtype=np.float32).reshape(9),
                    np.asarray([semantic_age_s], dtype=np.float32),
                )
            )
            expanded_state = np.concatenate((terminal_state, semantic_state)).astype(np.float32)
        else:
            raise ValueError(f"semantic terminal state dimension mismatch: {terminal_state.shape}")

        return {
            "ray_depth": np.asarray(terminal_observation["ray_depth"], dtype=np.float32),
            "state": expanded_state,
        }

    def _augment_terminal_observations(
        self,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        terminal_semantic_vectors: np.ndarray,
        terminal_semantic_age: np.ndarray,
    ) -> None:

        done_mask = np.asarray(dones, dtype=np.bool_).reshape(-1)
        for env_index in np.nonzero(done_mask)[0].tolist():
            terminal_observation = infos[env_index].get("terminal_observation")
            if terminal_observation is None:
                continue
            infos[env_index]["terminal_observation"] = self._augment_single_terminal_observation(
                terminal_observation,
                terminal_semantic_vectors[env_index],
                float(terminal_semantic_age[env_index]),
            )

    def _attach_semantic_info(self, infos: list[dict[str, Any]]) -> None:

        for env_index, info in enumerate(infos):
            info["semantic_text"] = str(self._semantic_texts[env_index])
            info["semantic_source"] = str(self._semantic_sources[env_index])
            info["semantic_age_s"] = float(self._semantic_age[env_index])
            info["semantic_parse_failed"] = bool(self._semantic_parse_failed[env_index])
            info["textsemantic_episode_llm_calls"] = int(self._episode_llm_calls[env_index])
            info["textsemantic_episode_llm_time_ms"] = float(self._episode_llm_time_ms[env_index])
            info["textsemantic_episode_llm_parse_failures"] = int(self._episode_llm_parse_failures[env_index])
            info["textsemantic_last_llm_raw"] = self._last_llm_raw_texts[env_index]
            info["textsemantic_last_llm_labels"] = self._last_llm_parsed_texts[env_index]
            info["textsemantic_last_llm_matrix"] = self._last_llm_matrix_texts[env_index]
            for key, value in self._last_qwen_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    info[f"textsemantic_{key}"] = float(value)

    def reset(self) -> dict[str, np.ndarray]:

        observation = super().reset()
        self._semantic_control_step = 0
        self._last_actions.fill(0.0)
        self._semantic_age.fill(0.0)
        self._last_qwen_metrics = {}
        self._reset_episode_llm_stats()
        self._rule_refresh(observation, source="rule_warm_start")
        return self._augment_observation(observation)

    def _print_terminal_episode(self, env_index: int, info: dict[str, Any], terminal_reward: float) -> None:

        with contextlib.redirect_stdout(io.StringIO()):
            super()._print_terminal_episode(env_index, info, terminal_reward)

        reason = int(info.get("base_terminal_reason", 0))
        if reason == 3:
            result = "success"
        elif reason == 4:
            result = "timeout"
        else:
            result = "False"
        episode = info.get("episode") or {}
        episode_return = float(episode.get("r", float(terminal_reward)))
        episode_id = int(info.get("base_episode_id", -1))
        route_id = int(info.get("base_route_id", -1))
        speed = float(info.get("inspection_current_speed_m_s", 0.0))
        reason_name = {1: "contact", 2: "bounds", 3: "success", 4: "timeout"}.get(reason, "unknown")
        calls = int(self._episode_llm_calls[env_index])
        def nullable_text(value: object) -> str:

            return "null" if value is None else str(value)

        if calls > 0:
            llm_time_s = f"{float(self._episode_llm_time_ms[env_index]) / 1000.0:.3f}"
            llm_raw = nullable_text(self._last_llm_raw_texts[env_index])
            llm_labels = nullable_text(self._last_llm_parsed_texts[env_index])
            llm_matrix = nullable_text(self._last_llm_matrix_texts[env_index])
        else:
            llm_time_s = "null"
            llm_raw = "null"
            llm_labels = "null"
            llm_matrix = "null"

        parse_failures = int(self._episode_llm_parse_failures[env_index])
        self._append_episode_llm_log(
            {
                "episode_id": episode_id,
                "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "transitions": self._console_transitions,
                "env_id": env_index,
                "route_id": route_id,
                "reason": reason_name,
                "result": result,
                "episode_return": f"{episode_return:.6f}",
                "llm_time_s": llm_time_s,
                "llm_calls": calls,
                "llm_raw": llm_raw,
                "llm_labels": llm_labels,
                "llm_matrix": llm_matrix,
                "parse_failures": parse_failures,
            }
        )
        print(
            (
                f"[EPISODE {episode_id:09d}] t={self._console_transitions:,} "
                f"env={env_index:02d} route={route_id} current={speed:.1f}m/s "
                f"reason={reason_name} reward={episode_return:.2f} result={result} "
                f"llm_labels={llm_labels} llm_matrix={llm_matrix} "
                f"llm_time_s={llm_time_s} llm_calls={calls} parse_failures={parse_failures}"
            ),
            flush=True,
        )
        self._reset_episode_llm_stats([env_index])

    def step_async(self, actions: np.ndarray) -> None:

        self._last_actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, 5).copy()
        super().step_async(actions)

    def step_wait(self):

        observation, rewards, dones, infos = super().step_wait()
        self._semantic_control_step += 1
        self._semantic_age += float(self.semantic_config.control_dt_s)
        terminal_semantic_vectors = self._semantic_vectors.copy()
        terminal_semantic_age = self._semantic_age.copy()
        self._augment_terminal_observations(infos, dones, terminal_semantic_vectors, terminal_semantic_age)

        if bool(np.asarray(dones, dtype=np.bool_).any()):
            base_state = self._base_state_for_policy(observation)
            ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
            for env_index in np.nonzero(np.asarray(dones, dtype=np.bool_))[0].tolist():
                tokens = rule_semantic_from_observation_base(base_state[env_index], ray_depth[env_index], None)
                self._semantic_vectors[env_index] = semantic_tokens_to_vector_base(tokens)
                self._semantic_texts[env_index] = semantic_text(tokens)
                self._semantic_sources[env_index] = "rule_warm_start"
                self._semantic_parse_failed[env_index] = False
                self._semantic_age[env_index] = 0.0

        self._maybe_refresh_semantic(observation)
        self._attach_semantic_info(infos)
        return self._augment_observation(observation), rewards, dones, infos


__all__ = [
    "DEFAULT_QWEN_BASE_TEXT",
    "DEFAULT_QWEN_LORA_TEXT",
    "SemanticVecEnvWrapper",
    "SemanticWrapperConfig",
]


DEFAULT_QWEN_BASE = os.environ.get(
    "CSPI_QWEN_BASE",
    str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-4B"),
)
DEFAULT_QWEN_LORA = os.environ.get(
    "CSPI_QWEN_LORA",
    str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-4B-SFT-LoRA"),
)

COMMIT_HOLD_LLM_LOG_FIELDS = (
    "episode_id",
    "timestamp",
    "transitions",
    "env_id",
    "route_id",
    "reason",
    "result",
    "episode_return",
    "llm_time_s",
    "llm_calls",
    "mean_qwen_call_s",
    "max_qwen_call_s",
    "qwen_deadline_s",
    "qwen_deadline_miss_count",
    "qwen_deadline_miss_rate",
    "llm_raw",
    "qwen_labels_raw",
    "llm_labels",
    "llm_matrix",
    "parse_failures",
    "hold_last_timeout_count",
    "hold_last_parse_fail_count",
    "safety_merge_count",
    "emergency_override_count",
    "hold_last_count",
    "qwen_compact_json_count",
    "rule_warmstart_count",
    "semantic_refresh_count",
    "semantic_source_counts",
    "mean_semantic_age",
    "max_semantic_age",
    "provider",
    "prompt_version",
)

DEBUG_LOG_FIELDS = (
    "refresh_index",
    "timestamp",
    "transitions",
    "env_id",
    "route_id",
    "source",
    "qwen_latency_ms",
    "qwen_latency_s",
    "qwen_deadline_missed",
    "semantic_age",
    "front_clearance",
    "left_clearance",
    "right_clearance",
    "up_clearance",
    "down_clearance",
    "forward_motion",
    "lateral_motion",
    "vertical_motion",
    "forward_action",
    "lateral_action",
    "vertical_action",
    "fossen_lateral",
    "fossen_vertical",
    "body_velocity_x_m_s",
    "body_velocity_y_m_s",
    "body_velocity_z_m_s",
    "qwen_raw",
    "qwen_labels_raw",
    "final_labels",
    "final_matrix",
)

ACTION_LABEL_ORDER = (
    "CRUISE",
    "ALIGN_GOAL",
    "AVOID_LEFT",
    "AVOID_RIGHT",
    "AVOID_UP",
    "AVOID_DOWN",
    "BRAKE",
    "REVERSE",
)


@dataclass
class CommitHoldWrapperConfig(SemanticWrapperConfig):

    qwen_base_model_path: str = DEFAULT_QWEN_BASE
    qwen_lora_adapter_path: str = DEFAULT_QWEN_LORA
    qwen_batch_size: int = 32
    enable_qwen: bool = True
    qwen_max_new_tokens: int = 64
    enable_safety_fallback: bool = True
    front_critical: float = 0.08
    front_low: float = 0.16
    side_critical: float = 0.08
    side_low: float = 0.14
    reverse_critical: float = 0.04
    direction_margin: float = 0.08
    qwen_deadline_s: float = 0.30
    debug_refresh_limit: int = 2000


class CommitHoldVecEnvWrapper(SemanticVecEnvWrapper):

    semantic_config: CommitHoldWrapperConfig

    def __init__(
        self,
        env,
        episode_log_path: Path,
        *,
        fast_variant: bool = True,
        semantic_config: CommitHoldWrapperConfig | None = None,
    ) -> None:
        self._semantic_histories: list[deque[FrameSnapshot]] = []
        super().__init__(
            env,
            episode_log_path=episode_log_path,
            fast_variant=fast_variant,
            semantic_config=semantic_config or CommitHoldWrapperConfig(),
        )
        self._semantic_histories = [deque(maxlen=4) for _ in range(self.num_envs)]
        self._debug_log_path = Path(self._episode_llm_log_path).with_name("semantic_refresh_debug.csv").resolve()
        self._debug_refresh_rows_written = 0
        self._ensure_debug_log_header()

    def _ensure_episode_llm_log_header(self) -> None:

        self._episode_llm_log_path = Path(self._episode_llm_log_path).with_name("semantic_episode_llm.csv").resolve()
        self._episode_llm_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._episode_llm_log_path.is_file() and self._episode_llm_log_path.stat().st_size > 0:
            with self._episode_llm_log_path.open("r", encoding="utf-8", newline="") as stream:
                actual = tuple(next(csv.reader(stream), ()))
            if actual != COMMIT_HOLD_LLM_LOG_FIELDS:
                raise RuntimeError(
                    "semantic episode LLM CSV header mismatch: "
                    f"actual={actual}, expected={COMMIT_HOLD_LLM_LOG_FIELDS}"
                )
            return
        with self._episode_llm_log_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=COMMIT_HOLD_LLM_LOG_FIELDS).writeheader()
            stream.flush()

    def _ensure_debug_log_header(self) -> None:

        if int(self.semantic_config.debug_refresh_limit) <= 0:
            return
        self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._debug_log_path.is_file() and self._debug_log_path.stat().st_size > 0:
            with self._debug_log_path.open("r", encoding="utf-8", newline="") as stream:
                actual = tuple(next(csv.reader(stream), ()))
            if actual != DEBUG_LOG_FIELDS:
                raise RuntimeError(
                    "semantic debug CSV header mismatch: "
                    f"actual={actual}, expected={DEBUG_LOG_FIELDS}"
                )
            return
        with self._debug_log_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=DEBUG_LOG_FIELDS).writeheader()
            stream.flush()

    def _append_episode_llm_log(self, record: dict[str, object]) -> None:

        missing = set(COMMIT_HOLD_LLM_LOG_FIELDS) - set(record)
        extra = set(record) - set(COMMIT_HOLD_LLM_LOG_FIELDS)
        if missing or extra:
            raise ValueError(f"semantic LLM CSV fields invalid: missing={sorted(missing)} extra={sorted(extra)}")
        with self._episode_llm_log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=COMMIT_HOLD_LLM_LOG_FIELDS).writerow(
                {field: record[field] for field in COMMIT_HOLD_LLM_LOG_FIELDS}
            )
            stream.flush()

    def _reset_episode_llm_stats(self, env_indices: np.ndarray | list[int] | None = None) -> None:

        super()._reset_episode_llm_stats(env_indices)
        if env_indices is None:
            indices = np.arange(self.num_envs, dtype=np.int64)
        else:
            indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)

        if not hasattr(self, "_episode_qwen_deadline_misses"):
            self._episode_qwen_deadline_misses = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_qwen_latency_max_ms = np.zeros(self.num_envs, dtype=np.float64)
            self._episode_hold_last_timeout_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_hold_last_parse_fail_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_safety_merge_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_emergency_override_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_qwen_success_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_rule_warmstart_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_semantic_refresh_count = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_semantic_age_sum = np.zeros(self.num_envs, dtype=np.float64)
            self._episode_semantic_age_max = np.zeros(self.num_envs, dtype=np.float64)
            self._episode_source_counters = [Counter() for _ in range(self.num_envs)]
            self._last_qwen_label_texts = np.full(self.num_envs, None, dtype=object)

        self._episode_qwen_deadline_misses[indices] = 0
        self._episode_qwen_latency_max_ms[indices] = 0.0
        self._episode_hold_last_timeout_count[indices] = 0
        self._episode_hold_last_parse_fail_count[indices] = 0
        self._episode_safety_merge_count[indices] = 0
        self._episode_emergency_override_count[indices] = 0
        self._episode_qwen_success_count[indices] = 0
        self._episode_rule_warmstart_count[indices] = 0
        self._episode_semantic_refresh_count[indices] = 0
        self._episode_semantic_age_sum[indices] = 0.0
        self._episode_semantic_age_max[indices] = 0.0
        self._last_qwen_label_texts[indices] = None
        for env_index in indices.tolist():
            self._episode_source_counters[int(env_index)].clear()

    def _append_debug_log(self, record: dict[str, object]) -> None:

        if int(self.semantic_config.debug_refresh_limit) <= 0:
            return
        if self._debug_refresh_rows_written >= int(self.semantic_config.debug_refresh_limit):
            return
        missing = set(DEBUG_LOG_FIELDS) - set(record)
        extra = set(record) - set(DEBUG_LOG_FIELDS)
        if missing or extra:
            raise ValueError(f"semantic debug CSV fields invalid: missing={sorted(missing)} extra={sorted(extra)}")
        with self._debug_log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=DEBUG_LOG_FIELDS).writerow(
                {field: record[field] for field in DEBUG_LOG_FIELDS}
            )
            stream.flush()
        self._debug_refresh_rows_written += 1

    def _qwen(self) -> QwenJsonSemanticBatcher:

        if self._qwen_batcher is None:
            cfg = self.semantic_config
            self._qwen_batcher = QwenJsonSemanticBatcher(
                cfg.qwen_base_model_path,
                cfg.qwen_lora_adapter_path,
                device=cfg.qwen_device,
                torch_dtype=cfg.qwen_torch_dtype,
                qwen_batch_size=cfg.qwen_batch_size,
                max_new_tokens=int(cfg.qwen_max_new_tokens),
            )
        return self._qwen_batcher

    def _seed_history_from_observation(self, observation: dict[str, np.ndarray]) -> None:

        base_state = self._base_state_for_policy(observation)
        ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
        for env_index in range(self.num_envs):
            snapshot = capture_frame_snapshot(base_state[env_index], ray_depth[env_index], self._last_actions[env_index])
            self._semantic_histories[env_index].clear()
            for _ in range(4):
                self._semantic_histories[env_index].append(snapshot)

    def _seed_history_for_env(self, observation: dict[str, np.ndarray], env_index: int) -> None:

        base_state = self._base_state_for_policy(observation)
        ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
        snapshot = capture_frame_snapshot(base_state[env_index], ray_depth[env_index], self._last_actions[env_index])
        self._semantic_histories[env_index].clear()
        for _ in range(4):
            self._semantic_histories[env_index].append(snapshot)

    def _capture_history_frame(self, observation: dict[str, np.ndarray]) -> None:

        base_state = self._base_state_for_policy(observation)
        ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
        for env_index in range(self.num_envs):
            self._semantic_histories[env_index].append(
                capture_frame_snapshot(base_state[env_index], ray_depth[env_index], self._last_actions[env_index])
            )

    def _history_for_env(self, env_index: int) -> list[FrameSnapshot]:

        history = list(self._semantic_histories[env_index])
        if not history:
            raise RuntimeError("semantic history is empty; reset should seed histories before Qwen refresh")
        while len(history) < 4:
            history.insert(0, history[0])
        return history[-4:]

    def _route_id_for_env(self, env_index: int) -> int:

        for obj in (getattr(self, "unwrapped", None), getattr(self, "env", None)):
            if obj is None:
                continue
            for name in ("_route_ids", "route_ids", "_route_id", "route_id"):
                value = getattr(obj, name, None)
                if value is None:
                    continue
                try:
                    if hasattr(value, "detach"):
                        value = value.detach().cpu().numpy()
                    array = np.asarray(value).reshape(-1)
                    if array.size > env_index:
                        return int(array[env_index])
                    if array.size == 1:
                        return int(array[0])
                except Exception:
                    continue
        return -1

    def _ordered_tokens(self, labels: set[str]) -> tuple[str, ...]:

        return tuple(label for label in ACTION_LABEL_ORDER if label in labels)

    def _resolve_direction_conflicts(self, labels: set[str], clear: dict[str, float]) -> set[str]:

        margin = float(self.semantic_config.direction_margin)
        resolved = set(labels)
        if "AVOID_LEFT" in resolved and "AVOID_RIGHT" in resolved:
            left = float(clear.get("left", 0.0))
            right = float(clear.get("right", 0.0))
            resolved.discard("AVOID_LEFT")
            resolved.discard("AVOID_RIGHT")
            if left > right + margin:
                resolved.add("AVOID_LEFT")
            elif right > left + margin:
                resolved.add("AVOID_RIGHT")
        if "AVOID_UP" in resolved and "AVOID_DOWN" in resolved:
            up = float(clear.get("up", 0.0))
            down = float(clear.get("down", 0.0))
            resolved.discard("AVOID_UP")
            resolved.discard("AVOID_DOWN")
            if up > down + margin:
                resolved.add("AVOID_UP")
            elif down > up + margin:
                resolved.add("AVOID_DOWN")
        return resolved

    def _add_freer_direction(self, labels: set[str], clear: dict[str, float], *, horizontal: bool, vertical: bool) -> None:

        margin = float(self.semantic_config.direction_margin)
        if horizontal:
            left = float(clear.get("left", 0.0))
            right = float(clear.get("right", 0.0))
            if left > right + margin:
                labels.add("AVOID_LEFT")
            elif right > left + margin:
                labels.add("AVOID_RIGHT")
        if vertical:
            up = float(clear.get("up", 0.0))
            down = float(clear.get("down", 0.0))
            if up > down + margin:
                labels.add("AVOID_UP")
            elif down > up + margin:
                labels.add("AVOID_DOWN")

    def _apply_safety_fallback(
        self,
        vector: np.ndarray,
        source: str,
        history: list[FrameSnapshot],
    ) -> tuple[np.ndarray, str, bool, bool]:

        if not bool(self.semantic_config.enable_safety_fallback):
            return np.asarray(vector, dtype=np.float32).reshape(9), source, False, False

        clear = history[-1].clearance
        front = float(clear.get("front", 0.0))
        labels = set(semantic_vector_to_tokens(vector))
        labels.discard("STOP_HOLD")
        original_labels = set(labels)

        if front <= float(self.semantic_config.front_critical):
            emergency_labels = {"ALIGN_GOAL", "BRAKE"}
            self._add_freer_direction(emergency_labels, clear, horizontal=True, vertical=True)
            if front <= float(self.semantic_config.reverse_critical):
                emergency_labels.add("REVERSE")
            emergency_labels = self._resolve_direction_conflicts(emergency_labels, clear)
            emergency_vector = semantic_tokens_to_vector(self._ordered_tokens(emergency_labels))
            return emergency_vector, f"{source}_emergency_override", False, True

        low_risk = front <= float(self.semantic_config.front_low)
        for blocked, opposite in (("left", "AVOID_LEFT"), ("right", "AVOID_RIGHT"), ("up", "AVOID_UP"), ("down", "AVOID_DOWN")):
            if float(clear.get(blocked, 1.0)) <= float(self.semantic_config.side_critical):
                labels.discard(opposite)
                low_risk = True

        if front <= float(self.semantic_config.front_low):
            labels.discard("CRUISE")
            labels.add("BRAKE")
            self._add_freer_direction(labels, clear, horizontal=True, vertical=True)

        if float(clear.get("left", 1.0)) <= float(self.semantic_config.side_low):
            labels.discard("AVOID_LEFT")
            if float(clear.get("right", 0.0)) > float(clear.get("left", 0.0)) + float(self.semantic_config.direction_margin):
                labels.add("AVOID_RIGHT")
            low_risk = True
        if float(clear.get("right", 1.0)) <= float(self.semantic_config.side_low):
            labels.discard("AVOID_RIGHT")
            if float(clear.get("left", 0.0)) > float(clear.get("right", 0.0)) + float(self.semantic_config.direction_margin):
                labels.add("AVOID_LEFT")
            low_risk = True
        if float(clear.get("up", 1.0)) <= float(self.semantic_config.side_low):
            labels.discard("AVOID_UP")
            if float(clear.get("down", 0.0)) > float(clear.get("up", 0.0)) + float(self.semantic_config.direction_margin):
                labels.add("AVOID_DOWN")
            low_risk = True
        if float(clear.get("down", 1.0)) <= float(self.semantic_config.side_low):
            labels.discard("AVOID_DOWN")
            if float(clear.get("up", 0.0)) > float(clear.get("down", 0.0)) + float(self.semantic_config.direction_margin):
                labels.add("AVOID_UP")
            low_risk = True

        labels = self._resolve_direction_conflicts(labels, clear)
        if not labels:
            labels.add("ALIGN_GOAL")
        changed = labels != original_labels
        if low_risk and changed:
            return semantic_tokens_to_vector(self._ordered_tokens(labels)), f"{source}_safety_merge", True, False
        return semantic_tokens_to_vector(self._ordered_tokens(labels)), source, False, False

    def _record_semantic_refresh(self, env_index: int, source: str, *, safety_merge: bool, emergency_override: bool) -> None:

        self._episode_semantic_refresh_count[env_index] += 1
        self._episode_source_counters[env_index][source] += 1
        if source.startswith("qwen_compact_json"):
            self._episode_qwen_success_count[env_index] += 1
        if source.startswith("rule_warmstart"):
            self._episode_rule_warmstart_count[env_index] += 1
        if source.startswith("hold_last_timeout"):
            self._episode_hold_last_timeout_count[env_index] += 1
        if source.startswith("hold_last_parse_fail"):
            self._episode_hold_last_parse_fail_count[env_index] += 1
        if safety_merge:
            self._episode_safety_merge_count[env_index] += 1
        if emergency_override:
            self._episode_emergency_override_count[env_index] += 1

    def _commit_semantic(
        self,
        env_index: int,
        vector: np.ndarray,
        source: str,
        *,
        parse_failed: bool,
        qwen_raw: object | None,
        qwen_labels_raw: object | None,
        reset_age: bool,
        safety_merge: bool,
        emergency_override: bool,
        qwen_latency_ms: float | None = None,
        qwen_deadline_missed: bool | None = None,
    ) -> None:

        final_vector = np.asarray(vector, dtype=np.float32).reshape(9)
        final_tokens = semantic_vector_to_tokens(final_vector)
        final_text = semantic_text(final_tokens)
        self._semantic_vectors[env_index] = final_vector
        self._semantic_texts[env_index] = final_text
        self._semantic_sources[env_index] = source
        self._semantic_parse_failed[env_index] = bool(parse_failed)
        if reset_age:
            self._semantic_age[env_index] = 0.0
        self._semantic_age[env_index] = min(float(self._semantic_age[env_index]), 1.0)
        self._episode_semantic_age_sum[env_index] += float(self._semantic_age[env_index])
        self._episode_semantic_age_max[env_index] = max(
            float(self._episode_semantic_age_max[env_index]),
            float(self._semantic_age[env_index]),
        )
        self._record_semantic_refresh(env_index, source, safety_merge=safety_merge, emergency_override=emergency_override)

        self._last_qwen_label_texts[env_index] = qwen_labels_raw
        self._last_llm_parsed_texts[env_index] = final_text
        self._last_llm_matrix_texts[env_index] = semantic_vector_to_matrix_text(final_vector)

        history = self._history_for_env(env_index)
        debug_fields = frame_debug_fields(history)
        if qwen_latency_ms is None or not np.isfinite(float(qwen_latency_ms)):
            qwen_latency_ms_text = ""
            qwen_latency_s_text = ""
        else:
            qwen_latency_ms_text = f"{float(qwen_latency_ms):.3f}"
            qwen_latency_s_text = f"{float(qwen_latency_ms) / 1000.0:.6f}"
        if qwen_deadline_missed is None:
            qwen_deadline_missed_text = ""
        else:
            qwen_deadline_missed_text = "1" if bool(qwen_deadline_missed) else "0"
        self._append_debug_log(
            {
                "refresh_index": self._debug_refresh_rows_written,
                "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "transitions": self._console_transitions,
                "env_id": env_index,
                "route_id": self._route_id_for_env(env_index),
                "source": source,
                "qwen_latency_ms": qwen_latency_ms_text,
                "qwen_latency_s": qwen_latency_s_text,
                "qwen_deadline_missed": qwen_deadline_missed_text,
                "semantic_age": f"{float(self._semantic_age[env_index]):.3f}",
                "front_clearance": f"{float(debug_fields['front_clearance']):.6f}",
                "left_clearance": f"{float(debug_fields['left_clearance']):.6f}",
                "right_clearance": f"{float(debug_fields['right_clearance']):.6f}",
                "up_clearance": f"{float(debug_fields['up_clearance']):.6f}",
                "down_clearance": f"{float(debug_fields['down_clearance']):.6f}",
                "forward_motion": debug_fields["forward_motion"],
                "lateral_motion": debug_fields["lateral_motion"],
                "vertical_motion": debug_fields["vertical_motion"],
                "forward_action": debug_fields["forward_action"],
                "lateral_action": debug_fields["lateral_action"],
                "vertical_action": debug_fields["vertical_action"],
                "fossen_lateral": debug_fields["fossen_lateral"],
                "fossen_vertical": debug_fields["fossen_vertical"],
                "body_velocity_x_m_s": f"{float(debug_fields['body_velocity_x_m_s']):.6f}",
                "body_velocity_y_m_s": f"{float(debug_fields['body_velocity_y_m_s']):.6f}",
                "body_velocity_z_m_s": f"{float(debug_fields['body_velocity_z_m_s']):.6f}",
                "qwen_raw": "null" if qwen_raw is None else str(qwen_raw),
                "qwen_labels_raw": "null" if qwen_labels_raw is None else str(qwen_labels_raw),
                "final_labels": final_text,
                "final_matrix": semantic_vector_to_matrix_text(final_vector),
            }
        )

    def _rule_refresh(self, observation: dict[str, np.ndarray], source: str) -> None:

        base_state = self._base_state_for_policy(observation)
        ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
        for env_index in range(self.num_envs):
            tokens = rule_semantic_from_observation(
                base_state[env_index],
                ray_depth[env_index],
                self._last_actions[env_index],
            )
            base_vector = semantic_tokens_to_vector(tokens)
            history = self._history_for_env(env_index)
            final_vector, final_source, safety_merge, emergency_override = self._apply_safety_fallback(
                base_vector,
                source,
                history,
            )
            self._commit_semantic(
                env_index,
                final_vector,
                final_source,
                parse_failed=False,
                qwen_raw=None,
                qwen_labels_raw=None,
                reset_age=True,
                safety_merge=safety_merge,
                emergency_override=emergency_override,
            )

    def _qwen_refresh(self, observation: dict[str, np.ndarray]) -> None:

        histories = [self._history_for_env(env_index) for env_index in range(self.num_envs)]
        raw_texts, metrics = self._qwen().predict_texts(histories)
        self._last_qwen_metrics = metrics
        prompt_latencies = metrics.get("qwen_prompt_latency_ms", [])
        for env_index, raw_text in enumerate(raw_texts):
            if isinstance(prompt_latencies, list) and env_index < len(prompt_latencies):
                latency_ms = float(prompt_latencies[env_index])
            else:
                latency_ms = float("nan")
            self._episode_llm_calls[env_index] += 1
            qwen_deadline_missed = False
            if np.isfinite(latency_ms):
                self._episode_llm_time_ms[env_index] += latency_ms
                self._last_llm_latency_ms[env_index] = latency_ms
                self._episode_qwen_latency_max_ms[env_index] = max(
                    float(self._episode_qwen_latency_max_ms[env_index]),
                    latency_ms,
                )
                qwen_deadline_missed = latency_ms / 1000.0 > float(self.semantic_config.qwen_deadline_s)
                if qwen_deadline_missed:
                    self._episode_qwen_deadline_misses[env_index] += 1
            self._last_llm_raw_texts[env_index] = str(raw_text)

            parsed = parse_semantic_json_text(raw_text)
            if parsed.valid:
                parsed_text = semantic_text(parsed.tokens)
                history = self._history_for_env(env_index)
                final_vector, final_source, safety_merge, emergency_override = self._apply_safety_fallback(
                    parsed.vector,
                    "qwen_compact_json",
                    history,
                )
                self._commit_semantic(
                    env_index,
                    final_vector,
                    final_source,
                    parse_failed=False,
                    qwen_raw=raw_text,
                    qwen_labels_raw=parsed_text,
                    reset_age=True,
                    safety_merge=safety_merge,
                    emergency_override=emergency_override,
                    qwen_latency_ms=latency_ms,
                    qwen_deadline_missed=qwen_deadline_missed,
                )
            else:
                self._episode_llm_parse_failures[env_index] += 1
                history = self._history_for_env(env_index)
                final_vector, final_source, safety_merge, emergency_override = self._apply_safety_fallback(
                    self._semantic_vectors[env_index],
                    "hold_last_parse_fail",
                    history,
                )
                self._commit_semantic(
                    env_index,
                    final_vector,
                    final_source,
                    parse_failed=True,
                    qwen_raw=raw_text,
                    qwen_labels_raw=None,
                    reset_age=bool(safety_merge or emergency_override),
                    safety_merge=safety_merge,
                    emergency_override=emergency_override,
                    qwen_latency_ms=latency_ms,
                    qwen_deadline_missed=qwen_deadline_missed,
                )

    def _maybe_refresh_semantic(self, observation: dict[str, np.ndarray]) -> None:

        cfg = self.semantic_config
        should_refresh = self._semantic_control_step % int(cfg.qwen_period_steps) == 0
        if not should_refresh:
            return
        self._capture_history_frame(observation)
        if (not bool(cfg.enable_qwen)) or int(self._console_transitions) < int(cfg.qwen_enable_after_transitions):
            self._rule_refresh(observation, source="rule_warmstart")
        else:
            self._qwen_refresh(observation)

    def reset(self) -> dict[str, np.ndarray]:

        observation = AuvSb3VecEnvWrapper.reset(self)
        self._semantic_control_step = 0
        self._last_actions.fill(0.0)
        self._semantic_age.fill(0.0)
        self._last_qwen_metrics = {}
        self._reset_episode_llm_stats()
        self._seed_history_from_observation(observation)
        self._rule_refresh(observation, source="rule_warmstart")
        return self._augment_observation(observation)

    def step_wait(self):

        observation, rewards, dones, infos = AuvSb3VecEnvWrapper.step_wait(self)
        self._semantic_control_step += 1
        self._semantic_age += float(self.semantic_config.control_dt_s)
        self._semantic_age = np.minimum(self._semantic_age, 1.0).astype(np.float32, copy=False)
        terminal_semantic_vectors = self._semantic_vectors.copy()
        terminal_semantic_age = self._semantic_age.copy()
        self._augment_terminal_observations(infos, dones, terminal_semantic_vectors, terminal_semantic_age)

        done_indices = np.nonzero(np.asarray(dones, dtype=np.bool_))[0].tolist()
        if done_indices:
            base_state = self._base_state_for_policy(observation)
            ray_depth = np.asarray(observation["ray_depth"], dtype=np.float32)
            for env_index in done_indices:
                tokens = rule_semantic_from_observation(base_state[env_index], ray_depth[env_index], None)
                self._semantic_vectors[env_index] = semantic_tokens_to_vector(tokens)
                self._semantic_texts[env_index] = semantic_text(tokens)
                self._semantic_sources[env_index] = "rule_warmstart"
                self._semantic_parse_failed[env_index] = False
                self._semantic_age[env_index] = 0.0
                self._seed_history_for_env(observation, env_index)

        self._maybe_refresh_semantic(observation)
        self._attach_semantic_info(infos)
        return self._augment_observation(observation), rewards, dones, infos

    def _attach_semantic_info(self, infos: list[dict[str, Any]]) -> None:

        super()._attach_semantic_info(infos)
        for env_index, info in enumerate(infos):
            info["semantic_text"] = str(self._semantic_texts[env_index])
            info["semantic_source"] = str(self._semantic_sources[env_index])
            info["semantic_age_s"] = float(self._semantic_age[env_index])
            info["semantic_parse_failed"] = bool(self._semantic_parse_failed[env_index])
            info["semantic_episode_llm_calls"] = int(self._episode_llm_calls[env_index])
            info["semantic_episode_llm_time_ms"] = float(self._episode_llm_time_ms[env_index])
            info["semantic_episode_llm_parse_failures"] = int(self._episode_llm_parse_failures[env_index])
            info["semantic_last_llm_raw"] = self._last_llm_raw_texts[env_index]
            info["semantic_last_qwen_labels_raw"] = self._last_qwen_label_texts[env_index]
            info["semantic_last_llm_labels"] = self._last_llm_parsed_texts[env_index]
            info["semantic_last_llm_matrix"] = self._last_llm_matrix_texts[env_index]
            info["semantic_safety_merge_count"] = int(self._episode_safety_merge_count[env_index])
            info["semantic_emergency_override_count"] = int(self._episode_emergency_override_count[env_index])
            info["semantic_deadline_miss_count"] = int(self._episode_qwen_deadline_misses[env_index])
            info["semantic_text"] = str(self._semantic_texts[env_index])
            info["semantic_source"] = str(self._semantic_sources[env_index])
            info["semantic_age_s"] = float(self._semantic_age[env_index])
            info["semantic_parse_failed"] = bool(self._semantic_parse_failed[env_index])
            info["semantic_episode_llm_calls"] = int(self._episode_llm_calls[env_index])
            info["semantic_episode_llm_time_ms"] = float(self._episode_llm_time_ms[env_index])
            info["semantic_episode_llm_parse_failures"] = int(self._episode_llm_parse_failures[env_index])
            info["semantic_last_llm_raw"] = self._last_llm_raw_texts[env_index]
            info["semantic_last_llm_labels"] = self._last_llm_parsed_texts[env_index]
            info["semantic_last_llm_matrix"] = self._last_llm_matrix_texts[env_index]

    def _print_terminal_episode(self, env_index: int, info: dict[str, Any], terminal_reward: float) -> None:

        reason = int(info.get("base_terminal_reason", 0))
        if reason == 3:
            result = "success"
        elif reason == 4:
            result = "timeout"
        else:
            result = "False"
        episode = info.get("episode") or {}
        episode_return = float(episode.get("r", float(terminal_reward)))
        episode_id = int(info.get("base_episode_id", -1))
        route_id = int(info.get("base_route_id", -1))
        speed = float(info.get("inspection_current_speed_m_s", 0.0))
        reason_name = {1: "contact", 2: "bounds", 3: "success", 4: "timeout"}.get(reason, "unknown")
        calls = int(self._episode_llm_calls[env_index])
        semantic_refreshes = int(self._episode_semantic_refresh_count[env_index])

        def nullable_text(value: object) -> str:
            return "null" if value is None else str(value)

        if calls > 0:
            llm_time_s = f"{float(self._episode_llm_time_ms[env_index]) / 1000.0:.3f}"
            mean_qwen_call_s = f"{float(self._episode_llm_time_ms[env_index]) / max(calls, 1) / 1000.0:.4f}"
            max_qwen_call_s = f"{float(self._episode_qwen_latency_max_ms[env_index]) / 1000.0:.4f}"
            llm_raw = nullable_text(self._last_llm_raw_texts[env_index])
            qwen_labels_raw = nullable_text(self._last_qwen_label_texts[env_index])
            llm_labels = nullable_text(self._last_llm_parsed_texts[env_index])
            llm_matrix = nullable_text(self._last_llm_matrix_texts[env_index])
        else:
            llm_time_s = "null"
            mean_qwen_call_s = "null"
            max_qwen_call_s = "null"
            llm_raw = "null"
            qwen_labels_raw = "null"
            llm_labels = "null"
            llm_matrix = "null"

        parse_failures = int(self._episode_llm_parse_failures[env_index])
        deadline_misses = int(self._episode_qwen_deadline_misses[env_index])
        deadline_rate = float(deadline_misses) / max(calls, 1) if calls > 0 else 0.0
        hold_last_timeout_count = int(self._episode_hold_last_timeout_count[env_index])
        hold_last_parse_fail_count = int(self._episode_hold_last_parse_fail_count[env_index])
        hold_last_count = hold_last_timeout_count + hold_last_parse_fail_count
        source_counts = ";".join(
            f"{key}={value}" for key, value in sorted(self._episode_source_counters[env_index].items())
        )
        mean_semantic_age = (
            float(self._episode_semantic_age_sum[env_index]) / max(semantic_refreshes, 1)
            if semantic_refreshes > 0
            else 0.0
        )
        self._append_episode_llm_log(
            {
                "episode_id": episode_id,
                "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "transitions": self._console_transitions,
                "env_id": env_index,
                "route_id": route_id,
                "reason": reason_name,
                "result": result,
                "episode_return": f"{episode_return:.6f}",
                "llm_time_s": llm_time_s,
                "llm_calls": calls,
                "mean_qwen_call_s": mean_qwen_call_s,
                "max_qwen_call_s": max_qwen_call_s,
                "qwen_deadline_s": f"{float(self.semantic_config.qwen_deadline_s):.3f}",
                "qwen_deadline_miss_count": deadline_misses,
                "qwen_deadline_miss_rate": f"{deadline_rate:.6f}",
                "llm_raw": llm_raw,
                "qwen_labels_raw": qwen_labels_raw,
                "llm_labels": llm_labels,
                "llm_matrix": llm_matrix,
                "parse_failures": parse_failures,
                "hold_last_timeout_count": hold_last_timeout_count,
                "hold_last_parse_fail_count": hold_last_parse_fail_count,
                "safety_merge_count": int(self._episode_safety_merge_count[env_index]),
                "emergency_override_count": int(self._episode_emergency_override_count[env_index]),
                "hold_last_count": hold_last_count,
                "qwen_compact_json_count": int(self._episode_qwen_success_count[env_index]),
                "rule_warmstart_count": int(self._episode_rule_warmstart_count[env_index]),
                "semantic_refresh_count": semantic_refreshes,
                "semantic_source_counts": source_counts,
                "mean_semantic_age": f"{mean_semantic_age:.4f}",
                "max_semantic_age": f"{float(self._episode_semantic_age_max[env_index]):.4f}",
                "provider": "qwen3-4b-semantic-lora",
                "prompt_version": "commit_hold_compact_json_online_safety",
            }
        )
        print(
            (
                f"[EPISODE {episode_id:09d}] t={self._console_transitions:,} "
                f"env={env_index:02d} route={route_id} current={speed:.1f}m/s "
                f"reason={reason_name} reward={episode_return:.2f} result={result} "
                f"llm_labels={llm_labels} llm_matrix={llm_matrix} "
                f"llm_time_s={llm_time_s} llm_calls={calls} mean_call_s={mean_qwen_call_s} "
                f"source_counts={source_counts or 'null'} safety_merge={int(self._episode_safety_merge_count[env_index])} "
                f"emergency={int(self._episode_emergency_override_count[env_index])} "
                f"deadline_miss={deadline_misses} parse_failures={parse_failures}"
            ),
            flush=True,
        )
        self._reset_episode_llm_stats([env_index])


__all__ = [
    "DEFAULT_QWEN_BASE",
    "DEFAULT_QWEN_LORA",
    "CommitHoldVecEnvWrapper",
    "CommitHoldWrapperConfig",
]
