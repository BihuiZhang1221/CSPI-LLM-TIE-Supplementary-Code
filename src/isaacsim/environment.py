"""Isaac Lab environment definitions for the BSA-D2 inspection tasks."""

from __future__ import annotations

import csv

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import datetime as dt
import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Mapping
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import numpy as np
import torch
from colorama import Fore, Style
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import isaaclab.sim
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.ray_caster import (
    MultiMeshRayCaster,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
    MultiMeshRayCasterCfg,
    patterns,
)
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_mul
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from src.isaacsim.dynamics import (
    ExtendedSafeDistanceFieldSpec,
    build_extended_safe_distance_field,
)


def _dvl_janus_pattern(cfg: "DvlJanusPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:

    tilt = math.radians(float(cfg.elevation_deg))
    horizontal = math.sin(tilt) / math.sqrt(2.0)
    vertical = -math.cos(tilt)
    directions = torch.tensor(
        [
            (horizontal, horizontal, vertical),
            (horizontal, -horizontal, vertical),
            (-horizontal, horizontal, vertical),
            (-horizontal, -horizontal, vertical),
        ],
        dtype=torch.float32,
        device=device,
    )
    starts = torch.zeros_like(directions)
    return starts, directions


@configclass
class DvlJanusPatternCfg(patterns.PatternBaseCfg):

    func: Callable = _dvl_janus_pattern
    elevation_deg: float = 22.5


class BsaD2RayEnv(DirectRLEnv):

    cfg: BsaD2RayEnvCfg

    _TERMINAL_NONE = 0
    _TERMINAL_CONTACT = 1
    _TERMINAL_BOUNDS = 2
    _TERMINAL_SUCCESS = 3
    _TERMINAL_TIMEOUT = 4

    def __init__(self, cfg: BsaD2RayEnvCfg, render_mode: str | None = None, **kwargs):
        self._configure_warp_cache(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = 5
        self._actions = torch.zeros(self.num_envs, action_dim, dtype=torch.float32, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._wrench_forces_b = torch.zeros(self.num_envs, 1, 3, dtype=torch.float32, device=self.device)
        self._wrench_torques_b = torch.zeros(self.num_envs, 1, 3, dtype=torch.float32, device=self.device)

        self._workspace_lower = torch.tensor(
            [axis[0] for axis in self.cfg.workspace_bounds], dtype=torch.float32, device=self.device
        )
        self._workspace_upper = torch.tensor(
            [axis[1] for axis in self.cfg.workspace_bounds], dtype=torch.float32, device=self.device
        )
        self._route_starts = torch.tensor(self.cfg.route_starts, dtype=torch.float32, device=self.device)
        self._route_goals = torch.tensor(self.cfg.route_goals, dtype=torch.float32, device=self.device)
        self._route_ids = self._make_balanced_fixed_route_ids(
            self.num_envs, self.cfg.active_route_ids
        ).to(device=self.device)
        env_origins = self._env_origins()
        self._goal_pos_w = self._route_goals[self._route_ids] + env_origins

        self._ray_depth_history = torch.ones(
            self.num_envs,
            int(self.cfg.ray_depth_history),
            int(self.cfg.ray_depth_height),
            int(self.cfg.ray_depth_width),
            dtype=torch.float32,
            device=self.device,
        )
        self._ray_history_seed_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self._contact_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_contact_point_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._terminal_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._terminal_episode_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._transition_route_ids = self._route_ids.clone()
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_goal_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._previous_dsafe = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._previous_dsafe_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_dsafe = torch.zeros_like(self._previous_dsafe)
        self._last_dsafe_valid = torch.zeros_like(self._previous_dsafe_valid)
        self._last_dsafe_delta = torch.zeros_like(self._previous_dsafe)
        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_min_ray_m = torch.full(
            (self.num_envs,), float(self.cfg.ray_depth_max_distance_m), dtype=torch.float32, device=self.device
        )

        self._safe_distance_bundle = self._build_safe_distance_field()
        self._safe_distance_runtime = self._safe_distance_bundle.to_torch(self.device)
        self._resume_signature = self._make_resume_signature()
        self.base_contract_hash = str(self._resume_signature["runtime_contract_hash"])
        self.base_geometry_hash = str(self._safe_distance_bundle.cache_id)

        self._initialize_trajectory_buffers()
        self.extras["base_route_id"] = self._transition_route_ids.clone()
        self.extras["base_terminal_reason"] = self._terminal_reason.clone()
        self.extras["base_episode_id"] = self._terminal_episode_ids.clone()

    @staticmethod
    def _configure_warp_cache(cfg: BsaD2RayEnvCfg) -> None:

        cache_path = Path(getattr(cfg, "warp_cache_path", Path(cfg.safe_distance_cache_dir).parent / "warp_cache"))
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["WARP_CACHE_PATH"] = str(cache_path)
        try:
            import warp as wp

            wp.config.kernel_cache_dir = str(cache_path)
        except Exception as exc:
            raise RuntimeError(f"cannot configure a writable Warp kernel cache: {cache_path}") from exc

    def _env_origins(self) -> torch.Tensor:
        origins = self.scene.env_origins
        if origins is None:
            return torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        return origins

    @staticmethod
    def _make_balanced_fixed_route_ids(num_envs: int, active_routes: Sequence[int]) -> torch.Tensor:

        routes = tuple(int(route_id) for route_id in active_routes)
        if routes != (0, 1, 2, 3, 4, 5, 6):
            raise ValueError(f"base active routes must be (0,1,2,3,4,5,6), got {routes}")
        base, extra = divmod(int(num_envs), len(routes))
        values: list[int] = []
        for slot, route_id in enumerate(routes):
            values.extend([route_id] * (base + (1 if slot < extra else 0)))
        return torch.tensor(values, dtype=torch.long)

    def _setup_scene(self) -> None:

        stage = self.scene.stage
        source_auv_path = self.cfg.auv_source_root_path
        source_auv_prim = stage.DefinePrim(source_auv_path, "Xform")
        source_auv_prim.GetReferences().AddReference(
            self.cfg.bsa_d2_usd_path, Sdf.Path(self.cfg.bsa_d2_reference_prim_path)
        )
        auv_xform = UsdGeom.Xformable(source_auv_prim)
        auv_xform.ClearXformOpOrder()
        auv_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        source_scene = stage.DefinePrim(self.cfg.avoid_x10_source_prim_path, "Xform")
        source_scene.GetReferences().AddReference(
            self.cfg.avoid_x10_usd_path, Sdf.Path(self.cfg.avoid_x10_reference_prim_path)
        )
        scene_xform = UsdGeom.Xformable(source_scene)
        scene_xform.ClearXformOpOrder()
        raw_min = self.cfg.avoid_x10_raw_scene_bbox_min
        raw_max = self.cfg.avoid_x10_raw_scene_bbox_max
        raw_bottom_center = Gf.Vec3d(
            0.5 * (raw_min[0] + raw_max[0]), 0.5 * (raw_min[1] + raw_max[1]), raw_min[2]
        )
        target_bottom_center = Gf.Vec3d(
            self.cfg.avoid_x10_target_center_xy[0],
            self.cfg.avoid_x10_target_center_xy[1],
            self.cfg.seabed_z_m,
        )
        scene_xform.AddTranslateOp(opSuffix="target_bottom_center").Set(target_bottom_center)
        scene_xform.AddScaleOp().Set(Gf.Vec3f(*(float(self.cfg.avoid_x10_scale),) * 3))
        scene_xform.AddTranslateOp(opSuffix="raw_bottom_center_to_origin").Set(-raw_bottom_center)

        for relative_path in self.cfg.avoid_x10_removed_prim_paths:
            normalized = str(relative_path).strip().strip("/")
            prim_path = f"{self.cfg.avoid_x10_source_prim_path}/{normalized}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"deletion-list prim not found; refusing to start with incorrect geometry: {prim_path}")
            prim.SetActive(False)

        pipe_root_path = f"{self.cfg.avoid_x10_source_prim_path}/pipe_colliders_x10"
        visual_root_path = f"{self.cfg.avoid_x10_source_prim_path}/visual_obstacles_x10"
        self._validate_scene_geometry(stage, pipe_root_path, visual_root_path)
        if self.cfg.hide_avoid_x10_visual_obstacles:
            self._deactivate_if_valid(stage, visual_root_path)
        if self.cfg.hide_auv_cad_visual:
            self._deactivate_if_valid(stage, f"{source_auv_path}/base_link/visuals/bsa_d2_visual")

        self._configure_physx_contract(stage, pipe_root_path)

        auv_cfg = RigidObjectCfg(
            prim_path=self.cfg.auv_prim_path,
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.cfg.route_starts[0],
                rot=(0.0, 0.0, 0.0, 1.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )
        self._auv = RigidObject(auv_cfg)
        self.scene.rigid_objects["auv"] = self._auv

        targets = self._raycast_targets()
        camera_cfg = MultiMeshRayCasterCameraCfg(
            prim_path=self.cfg.auv_prim_path,
            mesh_prim_paths=targets,
            update_period=0.0,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=self.cfg.ray_camera_offset_pos_m,
                rot=self.cfg.ray_camera_offset_rot_wxyz,
                convention="world",
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=float(self.cfg.ray_camera_focal_length),
                horizontal_aperture=float(self.cfg.ray_camera_horizontal_aperture),
                vertical_aperture=float(self.cfg.ray_camera_vertical_aperture),
                width=int(self.cfg.ray_depth_width),
                height=int(self.cfg.ray_depth_height),
            ),
            data_types=["distance_to_camera"],
            depth_clipping_behavior="max",
            max_distance=float(self.cfg.ray_depth_max_distance_m),
            update_mesh_ids=False,
            reference_meshes=True,
            debug_vis=False,
        )
        self._ray_camera = MultiMeshRayCasterCamera(camera_cfg)
        self.scene.sensors["base_ray_camera"] = self._ray_camera

        dvl_cfg = MultiMeshRayCasterCfg(
            prim_path=self.cfg.auv_prim_path,
            mesh_prim_paths=self._raycast_targets(),
            update_period=0.0,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=self.cfg.dvl_offset_pos_m),
            ray_alignment="base",
            pattern_cfg=DvlJanusPatternCfg(elevation_deg=float(self.cfg.dvl_elevation_deg)),
            max_distance=float(self.cfg.dvl_max_range_m),
            update_mesh_ids=False,
            reference_meshes=True,
            debug_vis=False,
        )
        self._dvl_raycaster = MultiMeshRayCaster(dvl_cfg)
        self.scene.sensors["base_dvl"] = self._dvl_raycaster

        contact_cfg = ContactSensorCfg(
            prim_path=self.cfg.auv_prim_path,
            update_period=0.0,
            history_length=0,
            track_pose=False,
            track_contact_points=False,
            track_friction_forces=False,
            max_contact_data_count_per_prim=64,
            filter_prim_paths_expr=self._contact_filter_paths(stage, pipe_root_path),
            debug_vis=False,
        )
        self._contact_sensor = ContactSensor(contact_cfg)
        self.scene.sensors["base_contact"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.85, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _raycast_targets(self) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:

        return [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/oil_rig_avoidance/pipe_colliders_x10",
                is_shared=True,
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/oil_rig_avoidance/seabed_8k",
                is_shared=True,
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            ),
        ]

    def _contact_filter_paths(self, stage, pipe_root_path: str) -> list[str]:

        pipe_prims = self._active_collision_prims(stage.GetPrimAtPath(pipe_root_path))
        if len(pipe_prims) != int(self.cfg.pipe_collider_count_expected):
            raise RuntimeError(
                "ContactView pipe-filter count does not match the valid-collider contract:"
                f"expected={self.cfg.pipe_collider_count_expected}, actual={len(pipe_prims)}"
            )
        seabed_root = stage.GetPrimAtPath(f"{self.cfg.avoid_x10_source_prim_path}/seabed_8k")
        seabed_prims = self._active_collision_prims(seabed_root)
        if not seabed_prims and seabed_root.HasAPI(UsdPhysics.CollisionAPI):
            seabed_prims = [seabed_root]
        if not seabed_prims:
            raise RuntimeError("ContactView cannot find the seabed_8k CollisionAPI.")

        source_prefix = "/World/envs/env_0/"
        regex_prefix = "/World/envs/env_.*/"
        patterns: list[str] = []
        for prim in [*pipe_prims, *seabed_prims]:
            source_path = prim.GetPath().pathString
            if not source_path.startswith(source_prefix):
                raise RuntimeError(f"base collider is not under source env_0: {source_path}")
            patterns.append(regex_prefix + source_path[len(source_prefix) :])
        unique_patterns = list(dict.fromkeys(patterns))
        if len(unique_patterns) != len(patterns):
            raise RuntimeError("base ContactView ContactView filter list contains duplicate colliders")
        return unique_patterns

    @staticmethod
    def _deactivate_if_valid(stage, prim_path: str) -> None:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            prim.SetActive(False)

    @staticmethod
    def _active_collision_prims(root_prim: Usd.Prim) -> list[Usd.Prim]:
        if not root_prim.IsValid():
            return []
        return [
            prim
            for prim in Usd.PrimRange(root_prim)
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]

    def _validate_scene_geometry(self, stage, pipe_root_path: str, visual_root_path: str) -> None:
        pipe_prims = self._active_collision_prims(stage.GetPrimAtPath(pipe_root_path))
        if len(pipe_prims) != int(self.cfg.pipe_collider_count_expected):
            raise RuntimeError(
                "base valid pipe-collider count does not match the deletion contract: "
                f"expected={self.cfg.pipe_collider_count_expected}, actual={len(pipe_prims)}"
            )
        visual_root = stage.GetPrimAtPath(visual_root_path)
        visual_mesh_count = sum(
            1 for prim in Usd.PrimRange(visual_root) if prim.IsActive() and prim.IsA(UsdGeom.Mesh)
        )
        if visual_mesh_count != int(self.cfg.visual_obstacle_mesh_count_expected):
            raise RuntimeError(
                "base visible obstacle count does not match the deletion contract: "
                f"expected={self.cfg.visual_obstacle_mesh_count_expected}, actual={visual_mesh_count}"
            )

    def _configure_physx_contract(self, stage, pipe_root_path: str) -> None:

        body_prim = stage.GetPrimAtPath(self.cfg.auv_source_body_path)
        if not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"base AUV base_link is not a valid rigid body: {self.cfg.auv_source_body_path}")
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
        physx_body.CreateLinearDampingAttr().Set(float(self.cfg.linear_damping))
        physx_body.CreateAngularDampingAttr().Set(float(self.cfg.angular_damping))
        physx_body.CreateEnableCCDAttr().Set(False)
        physx_body.CreateEnableSpeculativeCCDAttr().Set(False)
        physx_body.CreateSolverPositionIterationCountAttr().Set(
            int(self.cfg.solver_position_iteration_count)
        )
        physx_body.CreateSolverVelocityIterationCountAttr().Set(
            int(self.cfg.solver_velocity_iteration_count)
        )

        collision_root = stage.GetPrimAtPath(
            f"{self.cfg.auv_source_body_path}/{self.cfg.auv_collision_root_relative_path}"
        )
        auv_colliders = self._active_collision_prims(collision_root)
        names = tuple(sorted(prim.GetName() for prim in auv_colliders))
        expected = tuple(sorted(self.cfg.expected_auv_collision_names))
        if names != expected:
            raise RuntimeError(f"AUV three-collider contract mismatch: expected={expected}, actual={names}")
        for prim in auv_colliders:
            self._set_collision_shell(prim)

        pipe_prims = self._active_collision_prims(stage.GetPrimAtPath(pipe_root_path))
        for prim in pipe_prims:
            self._set_collision_shell(prim)
        seabed_root = stage.GetPrimAtPath(f"{self.cfg.avoid_x10_source_prim_path}/seabed_8k")
        seabed_colliders = self._active_collision_prims(seabed_root)
        if not seabed_colliders and seabed_root.HasAPI(UsdPhysics.CollisionAPI):
            seabed_colliders = [seabed_root]
        if not seabed_colliders:
            raise RuntimeError("seabed_8k CollisionAPI not found; the hard-contact contract cannot hold.")
        for prim in seabed_colliders:
            self._set_collision_shell(prim)

        sim_utils.activate_contact_sensors(self.cfg.auv_source_body_path, threshold=0.0, stage=stage)

    def _set_collision_shell(self, prim: Usd.Prim) -> None:
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr().Set(float(self.cfg.contact_offset_m))
        api.CreateRestOffsetAttr().Set(float(self.cfg.rest_offset_m))

    def _build_safe_distance_field(self):
        bounds = self.cfg.workspace_bounds
        spec = ExtendedSafeDistanceFieldSpec(
            lower=tuple(float(axis[0]) for axis in bounds),
            upper=tuple(float(axis[1]) for axis in bounds),
            resolution_m=float(self.cfg.safe_distance_resolution_m),
            clearance_m=float(self.cfg.safe_distance_core_clearance_m),
            extension_clearance_m=float(self.cfg.safe_distance_extension_clearance_m),
            active_route_ids=tuple(int(v) for v in self.cfg.active_route_ids),
        )
        starts = {route: self.cfg.route_starts[route] for route in self.cfg.active_route_ids}
        goals = {route: self.cfg.route_goals[route] for route in self.cfg.active_route_ids}
        origin0 = self._env_origins()[0].detach().cpu().tolist()
        scene_config = {
            "removed_prim_paths": tuple(self.cfg.avoid_x10_removed_prim_paths),
            "pipe_collider_count": int(self.cfg.pipe_collider_count_expected),
            "visual_mesh_count": int(self.cfg.visual_obstacle_mesh_count_expected),
            "map_scale": float(self.cfg.avoid_x10_scale),
            "workspace_bounds": tuple(tuple(float(v) for v in axis) for axis in bounds),
            "route_starts": tuple(self.cfg.route_starts),
            "route_goals": tuple(self.cfg.route_goals),
            "seabed_prim": "seabed_8k",
        }
        return build_extended_safe_distance_field(
            stage=self.scene.stage,
            collider_root_path=f"{self.cfg.avoid_x10_source_prim_path}/pipe_colliders_x10",
            scene_path=self.cfg.avoid_x10_usd_path,
            spec=spec,
            scene_config=scene_config,
            cache_dir=self.cfg.safe_distance_cache_dir,
            starts_by_route=starts,
            goals_by_route=goals,
            env_origin=origin0,
            diagnostics_dir=self.cfg.safe_distance_preflight_dir,
        )

    def _make_resume_signature(self) -> dict[str, object]:
        field_path = (
            Path(self.cfg.safe_distance_cache_dir)
            / f"base_safe_distance_field_{self._safe_distance_bundle.cache_id}.npz"
        )
        if not field_path.is_file():
            raise FileNotFoundError(f"safe distance field cache not found: {field_path}")
        field_digest = hashlib.sha256()
        with field_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                field_digest.update(chunk)
        geometry = {
            "safe_distance_cache_id": self._safe_distance_bundle.cache_id,
            "exact_mesh_digest": self._safe_distance_bundle.exact_mesh_digest,
            "safe_distance_field_sha256": field_digest.hexdigest(),
            "pipe_collider_count": int(self.cfg.pipe_collider_count_expected),
            "removed_prim_paths": list(self.cfg.avoid_x10_removed_prim_paths),
        }
        payload = {
            "static_contract": RAY_STATIC_CONTRACT,
            "static_contract_hash": RAY_STATIC_CONTRACT_HASH,
            "geometry": geometry,
        }
        payload["runtime_contract_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    def get_base_resume_signature(self, indices=None) -> dict[str, object]:

        del indices
        return json.loads(json.dumps(self._resume_signature, sort_keys=True))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 5):
            raise ValueError(f"base action must be [{self.num_envs},5], got {tuple(actions.shape)}")
        self._actions = actions.to(dtype=torch.float32).clamp(-1.0, 1.0)

    def _apply_action(self) -> None:

        self._latch_raw_contacts(armed_only=True)
        self._contact_armed[:] = True

        action = self._actions
        self._wrench_forces_b[:, 0, 0] = -float(self.cfg.force_limits_n[0]) * action[:, 0]
        self._wrench_forces_b[:, 0, 1] = -float(self.cfg.force_limits_n[1]) * action[:, 1]
        self._wrench_forces_b[:, 0, 2] = float(self.cfg.force_limits_n[2]) * action[:, 2]

        roll, _pitch, _yaw = euler_xyz_from_quat(self._auv.data.root_quat_w)
        omega_x = self._auv.data.root_ang_vel_b[:, 0]
        roll_torque = (
            -float(self.cfg.roll_pd_kp_nm_rad) * roll
            - float(self.cfg.roll_pd_kd_nm_s_rad) * omega_x
        ).clamp(-float(self.cfg.roll_pd_limit_nm), float(self.cfg.roll_pd_limit_nm))
        self._wrench_torques_b[:, 0, 0] = roll_torque
        self._wrench_torques_b[:, 0, 1] = float(self.cfg.torque_limits_nm[0]) * action[:, 3]
        self._wrench_torques_b[:, 0, 2] = float(self.cfg.torque_limits_nm[1]) * action[:, 4]
        self._auv.permanent_wrench_composer.set_forces_and_torques(
            forces=self._wrench_forces_b,
            torques=self._wrench_torques_b,
            body_ids=[0],
            is_global=False,
        )

    def _latch_raw_contacts(self, *, armed_only: bool) -> None:

        _forces, _points, _normals, _separations, buffer_count, _starts = (
            self._contact_sensor.contact_physx_view.get_contact_data(dt=float(self.physics_dt))
        )
        if buffer_count.numel() % self.num_envs != 0:
            raise RuntimeError(
                "contact buffer cannot be regrouped per env:"
                f"numel={buffer_count.numel()}, num_envs={self.num_envs}"
            )
        counts = buffer_count.reshape(self.num_envs, -1).to(torch.long).sum(dim=1)
        valid_counts = counts
        if armed_only:
            valid_counts = torch.where(self._contact_armed, counts, torch.zeros_like(counts))
        contacted = valid_counts > 0
        self._contact_latched |= contacted
        self._last_contact_point_count = torch.maximum(
            self._last_contact_point_count, valid_counts
        )

    def _current_depth_frame(self) -> torch.Tensor:
        raw = self._ray_camera.data.output["distance_to_camera"]
        if raw.shape != (
            self.num_envs,
            int(self.cfg.ray_depth_height),
            int(self.cfg.ray_depth_width),
            1,
        ):
            raise RuntimeError(f"base ray camera shape mismatch: {tuple(raw.shape)}")
        depth = torch.nan_to_num(
            raw[..., 0],
            nan=float(self.cfg.ray_depth_max_distance_m),
            posinf=float(self.cfg.ray_depth_max_distance_m),
            neginf=0.0,
        ).clamp(0.0, float(self.cfg.ray_depth_max_distance_m))
        return torch.log1p(depth) / math.log1p(float(self.cfg.ray_depth_max_distance_m))

    def _read_dvl(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hits_w = self._dvl_raycaster.data.ray_hits_w
        if hits_w.shape != (self.num_envs, 4, 3):
            raise RuntimeError(f"base DVL ray shape mismatch: {tuple(hits_w.shape)}")
        origin_w = self._dvl_raycaster.data.pos_w.unsqueeze(1)
        finite = torch.isfinite(hits_w).all(dim=-1)
        ranges = torch.linalg.norm(hits_w - origin_w, dim=-1)
        hit = finite & (ranges >= float(self.cfg.dvl_min_range_m)) & (
            ranges <= float(self.cfg.dvl_max_range_m)
        )
        ranges = torch.where(hit, ranges, torch.full_like(ranges, float(self.cfg.dvl_max_range_m)))
        ranges_norm = torch.log1p(ranges) / math.log1p(float(self.cfg.dvl_max_range_m))

        physical_velocity = self._auv.data.root_lin_vel_b
        semantic_velocity = torch.stack(
            (-physical_velocity[:, 0], -physical_velocity[:, 1], physical_velocity[:, 2]), dim=1
        )
        missing = (~hit).sum(dim=1)
        dropout = missing >= int(self.cfg.dvl_dropout_missing_beams)
        semantic_velocity = torch.where(dropout.unsqueeze(1), torch.zeros_like(semantic_velocity), semantic_velocity)
        scales = torch.tensor(self.cfg.dvl_velocity_scales_m_s, dtype=torch.float32, device=self.device)
        velocity_norm = (semantic_velocity / scales.unsqueeze(0)).clamp(-1.0, 1.0)
        return velocity_norm, ranges_norm.clamp(0.0, 1.0), hit.to(torch.float32)

    def _build_state_observation(self) -> torch.Tensor:
        velocity, dvl_ranges, dvl_hits = self._read_dvl()
        root_quat = self._auv.data.root_quat_w
        semantic_offset = torch.zeros_like(root_quat)
        semantic_offset[:, 3] = 1.0
        semantic_quat = quat_mul(root_quat, semantic_offset)
        semantic_quat = semantic_quat / torch.linalg.norm(semantic_quat, dim=1, keepdim=True).clamp_min(1.0e-8)
        semantic_quat = torch.where(semantic_quat[:, :1] < 0.0, -semantic_quat, semantic_quat)

        omega = self._auv.data.root_ang_vel_b
        semantic_omega = torch.stack((-omega[:, 0], omega[:, 1], omega[:, 2]), dim=1)
        semantic_omega = (
            semantic_omega / float(self.cfg.imu_angular_rate_scale_rad_s)
        ).clamp(-1.0, 1.0)

        root_local = self._auv.data.root_pos_w - self._env_origins()
        depth_m = (float(self.cfg.water_surface_z_m) - root_local[:, 2]).clamp_min(0.0)
        pressure_pa = (
            float(self.cfg.atmosphere_pressure_pa)
            + float(self.cfg.water_density_kg_m3) * float(self.cfg.pressure_gravity_m_s2) * depth_m
        )
        recovered_depth = (pressure_pa - float(self.cfg.atmosphere_pressure_pa)) / (
            float(self.cfg.water_density_kg_m3) * float(self.cfg.pressure_gravity_m_s2)
        )
        pressure_depth_norm = (
            (recovered_depth - float(self.cfg.pressure_depth_center_m))
            / float(self.cfg.pressure_depth_scale_m)
        ).clamp(-1.0, 1.0)

        goal_error_w = self._goal_pos_w - self._auv.data.root_pos_w
        goal_distance = torch.linalg.norm(goal_error_w, dim=1)
        goal_error_physical_b = quat_apply_inverse(root_quat, goal_error_w)
        goal_error_semantic_b = torch.stack(
            (-goal_error_physical_b[:, 0], -goal_error_physical_b[:, 1], goal_error_physical_b[:, 2]), dim=1
        )
        goal_direction = goal_error_semantic_b / goal_distance.unsqueeze(1).clamp_min(1.0e-8)
        goal_direction = torch.where(
            (goal_distance > 1.0e-8).unsqueeze(1), goal_direction, torch.zeros_like(goal_direction)
        )
        goal_distance_norm = (goal_distance / float(self.cfg.goal_distance_scale_m)).clamp(0.0, 1.0)

        state = torch.cat(
            (
                velocity,
                dvl_ranges,
                dvl_hits,
                semantic_quat,
                semantic_omega,
                pressure_depth_norm.unsqueeze(1),
                goal_direction,
                goal_distance_norm.unsqueeze(1),
                self._previous_actions,
            ),
            dim=1,
        )
        if state.shape != (self.num_envs, 28):
            raise RuntimeError(f"state concatenation dimension mismatch: {tuple(state.shape)}")
        if not torch.isfinite(state).all():
            raise RuntimeError("base state observation contains NaN or Inf")
        return state.to(torch.float32)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        newest = self._current_depth_frame()
        pending = self._ray_history_seed_pending
        normal = ~pending
        if normal.any():
            self._ray_depth_history[normal, :-1] = self._ray_depth_history[normal, 1:].clone()
            self._ray_depth_history[normal, -1] = newest[normal]
        if pending.any():
            self._ray_depth_history[pending] = newest[pending].unsqueeze(1).expand(-1, 4, -1, -1)
            self._ray_history_seed_pending[pending] = False
        if not torch.isfinite(self._ray_depth_history).all():
            raise RuntimeError("base ray-depth history contains NaN or Inf")
        min_depth = torch.amin(
            torch.expm1(self._ray_depth_history[:, -1] * math.log1p(float(self.cfg.ray_depth_max_distance_m))),
            dim=(1, 2),
        )
        self._episode_min_ray_m = torch.minimum(self._episode_min_ray_m, min_depth)
        policy = {"ray_depth": self._ray_depth_history, "state": self._build_state_observation()}
        return {"policy": policy}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._latch_raw_contacts(armed_only=True)
        root_pos_w = self._auv.data.root_pos_w
        root_local = root_pos_w - self._env_origins()
        self._out_of_bounds = torch.any(
            (root_local < self._workspace_lower.unsqueeze(0))
            | (root_local > self._workspace_upper.unsqueeze(0)),
            dim=1,
        )
        self._last_goal_distance = torch.linalg.norm(self._goal_pos_w - root_pos_w, dim=1)
        inside_goal = self._last_goal_distance <= float(self.cfg.goal_radius_m)
        self._goal_reached = inside_goal & ~self._contact_latched & ~self._out_of_bounds
        reached_limit = self.episode_length_buf >= self.max_episode_length

        self._time_out = reached_limit & ~self._contact_latched & ~self._out_of_bounds & ~self._goal_reached
        terminated = self._contact_latched | self._out_of_bounds | self._goal_reached
        truncated = self._time_out
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        reason = torch.where(self._time_out, torch.full_like(reason, self._TERMINAL_TIMEOUT), reason)
        reason = torch.where(self._goal_reached, torch.full_like(reason, self._TERMINAL_SUCCESS), reason)
        reason = torch.where(self._out_of_bounds, torch.full_like(reason, self._TERMINAL_BOUNDS), reason)
        reason = torch.where(self._contact_latched, torch.full_like(reason, self._TERMINAL_CONTACT), reason)
        self._terminal_reason = reason

        self._terminal_episode_ids.fill_(-1)
        done_env_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            done_env_ids = torch.sort(done_env_ids).values
            first_episode_id = int(self._plot_serial)
            assigned_ids = torch.arange(
                first_episode_id,
                first_episode_id + done_env_ids.numel(),
                dtype=torch.long,
                device=self.device,
            )
            self._terminal_episode_ids[done_env_ids] = assigned_ids
            self._plot_serial += int(done_env_ids.numel())

        self._transition_route_ids = self._route_ids.clone()
        self.extras["base_route_id"] = self._transition_route_ids.clone()
        self.extras["base_terminal_reason"] = reason.clone()
        self.extras["base_contact_point_count"] = self._last_contact_point_count.clone()
        self.extras["base_episode_id"] = self._terminal_episode_ids.clone()
        self._record_trajectory_points(root_local, terminated | truncated)
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        root_local = self._auv.data.root_pos_w - self._env_origins()
        dsafe, valid = self._safe_distance_runtime.lookup(root_local, self._route_ids)
        delta_valid = valid & self._previous_dsafe_valid
        delta = torch.where(delta_valid, self._previous_dsafe - dsafe, torch.zeros_like(dsafe))
        delta = delta.clamp(
            -float(self.cfg.safe_distance_delta_cap_m), float(self.cfg.safe_distance_delta_cap_m)
        )
        process = (
            float(self.cfg.safe_distance_progress_scale) * delta
            + float(self.cfg.time_cost)
            - float(self.cfg.action_cost_scale) * torch.sum(torch.square(self._actions), dim=1)
            - float(self.cfg.action_delta_cost_scale)
            * torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        )
        terminal = self._terminal_reason != self._TERMINAL_NONE
        reward = torch.where(terminal, torch.zeros_like(process), process)
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_CONTACT,
            torch.full_like(reward, float(self.cfg.contact_penalty)),
            reward,
        )
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_BOUNDS,
            torch.full_like(reward, float(self.cfg.out_of_bounds_penalty)),
            reward,
        )
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_SUCCESS,
            torch.full_like(reward, float(self.cfg.success_reward)),
            reward,
        )
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_TIMEOUT,
            torch.full_like(reward, float(self.cfg.timeout_penalty)),
            reward,
        )

        self._last_dsafe = dsafe.detach()
        self._last_dsafe_valid = valid.detach()
        self._last_dsafe_delta = delta.detach()
        self._previous_dsafe = dsafe.detach()
        self._previous_dsafe_valid = valid.detach()
        self._previous_actions = self._actions.detach().clone()
        self._episode_return += reward
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if hasattr(self, "_terminal_reason"):
            for env_id in sorted(env_ids.detach().cpu().tolist()):
                if int(self._terminal_reason[env_id].item()) != self._TERMINAL_NONE:
                    episode_id = int(self._terminal_episode_ids[env_id].item())
                    if episode_id < 0:
                        raise RuntimeError(f"base terminated env={env_id} has no unified episode ID")
                    self._save_episode_trajectory_plot(env_id, episode_id)
        super()._reset_idx(env_ids)

        route_ids = self._route_ids[env_ids]
        starts_local = self._route_starts[route_ids]
        goals_local = self._route_goals[route_ids]
        positions_w = starts_local + self._env_origins()[env_ids]
        route_vector = goals_local - starts_local
        heading = torch.atan2(route_vector[:, 1], route_vector[:, 0])
        physical_yaw = heading + math.pi
        quaternion = torch.zeros(env_ids.numel(), 4, dtype=torch.float32, device=self.device)
        quaternion[:, 0] = torch.cos(0.5 * physical_yaw)
        quaternion[:, 3] = torch.sin(0.5 * physical_yaw)
        self._auv.write_root_pose_to_sim(torch.cat((positions_w, quaternion), dim=1), env_ids=env_ids)
        self._auv.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, dtype=torch.float32, device=self.device), env_ids=env_ids
        )
        self._goal_pos_w[env_ids] = goals_local + self._env_origins()[env_ids]

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._wrench_forces_b[env_ids] = 0.0
        self._wrench_torques_b[env_ids] = 0.0
        self._auv.permanent_wrench_composer.set_forces_and_torques(
            forces=self._wrench_forces_b[env_ids],
            torques=self._wrench_torques_b[env_ids],
            body_ids=[0],
            env_ids=env_ids,
            is_global=False,
        )
        self._contact_latched[env_ids] = False
        self._contact_armed[env_ids] = False
        self._last_contact_point_count[env_ids] = 0
        self._terminal_reason[env_ids] = self._TERMINAL_NONE
        self._goal_reached[env_ids] = False
        self._out_of_bounds[env_ids] = False
        self._time_out[env_ids] = False
        self._ray_history_seed_pending[env_ids] = True
        self._episode_return[env_ids] = 0.0
        self._episode_min_ray_m[env_ids] = float(self.cfg.ray_depth_max_distance_m)

        initial_dsafe, initial_valid = self._safe_distance_runtime.lookup(starts_local, route_ids)
        self._previous_dsafe[env_ids] = initial_dsafe
        self._previous_dsafe_valid[env_ids] = initial_valid
        self._last_dsafe[env_ids] = initial_dsafe
        self._last_dsafe_valid[env_ids] = initial_valid
        self._last_dsafe_delta[env_ids] = 0.0
        if not bool(torch.all(initial_valid)):
            bad_routes = route_ids[~initial_valid].detach().cpu().tolist()
            raise RuntimeError(f"reset start Dsafe invalid; refusing to train: routes={bad_routes}")
        self._reset_trajectory_buffers(env_ids, starts_local)

    def _initialize_trajectory_buffers(self) -> None:
        stride = max(int(self.cfg.episode_route_plot_stride), 1)
        capacity = int(math.ceil(self.max_episode_length / stride)) + 3
        self._trajectory_xyz = torch.zeros(
            self.num_envs, capacity, 3, dtype=torch.float32, device=self.device
        )
        self._trajectory_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._plot_serial = 0
        self._trajectory_obstacle_rgba_xy = self._build_trajectory_obstacle_projection()
        plot_mode = str(self.cfg.episode_route_plot_mode).strip().lower()
        if plot_mode not in {"training", "validation"}:
            raise ValueError(
                "episode_route_plot_mode must be either 'training' or 'validation'; "
                f"got {self.cfg.episode_route_plot_mode!r}"
            )
        validation_checkpoint = int(self.cfg.validation_checkpoint_transitions)
        if plot_mode == "validation" and validation_checkpoint < 0:
            raise ValueError(
                "base validation plotting requires validation_checkpoint_transitions >= 0"
            )
        self._episode_route_plot_mode = plot_mode
        self._validation_checkpoint_transitions = validation_checkpoint
        self._validation_plotted_routes: set[int] = set()

        output_dir = Path(self.cfg.episode_route_plot_dir)
        if bool(self.cfg.save_episode_route_plots):
            output_dir.mkdir(parents=True, exist_ok=True)
            if plot_mode == "training":
                pattern = re.compile(r"episode_(\d+)\.png$")
                existing = []
                for path in output_dir.glob("base_route_*_episode_*.png"):
                    match = pattern.search(path.name)
                    if match:
                        existing.append(int(match.group(1)))
                self._plot_serial = max(existing, default=-1) + 1

    def _build_trajectory_obstacle_projection(self) -> np.ndarray:

        bundle = self._safe_distance_bundle
        forbidden = np.asarray(bundle.core_forbidden, dtype=bool)
        clearance = np.asarray(bundle.clearance_m, dtype=np.float32)
        lower = np.asarray(bundle.spec.lower, dtype=np.float64)
        upper = np.asarray(bundle.spec.upper, dtype=np.float64)
        resolution = float(bundle.spec.resolution_m)
        core_clearance = float(bundle.spec.clearance_m)
        coordinates = [
            lower[axis] + np.arange(forbidden.shape[axis], dtype=np.float64) * resolution
            for axis in range(3)
        ]
        interior_indices = [
            np.flatnonzero(
                np.minimum(coordinates[axis] - lower[axis], upper[axis] - coordinates[axis])
                >= core_clearance - 1.0e-5
            )
            for axis in range(3)
        ]
        if any(indices.size == 0 for indices in interior_indices):
            raise RuntimeError("trajectory plot cannot find the interior grid after removing the workspace bounds.")

        selection = np.ix_(*interior_indices)
        interior_forbidden = forbidden[selection]
        interior_clearance = clearance[selection]
        safety_local = np.any(interior_forbidden, axis=2)
        physical_local = np.any(interior_clearance <= 0.5 * resolution + 1.0e-6, axis=2)

        safety_xy = np.zeros(forbidden.shape[:2], dtype=bool)
        physical_xy = np.zeros_like(safety_xy)
        xy_selection = np.ix_(interior_indices[0], interior_indices[1])
        safety_xy[xy_selection] = safety_local
        physical_xy[xy_selection] = physical_local
        if not bool(np.any(physical_xy)):
            raise RuntimeError("trajectory plot exact obstacle projection is empty.")

        rgba = np.zeros((forbidden.shape[1], forbidden.shape[0], 4), dtype=np.uint8)
        rgba[safety_xy.T] = np.asarray([184, 197, 203, 105], dtype=np.uint8)
        rgba[physical_xy.T] = np.asarray([48, 58, 64, 235], dtype=np.uint8)
        return rgba

    def _reset_trajectory_buffers(self, env_ids: torch.Tensor, starts_local: torch.Tensor) -> None:
        self._trajectory_count[env_ids] = 1
        self._trajectory_xyz[env_ids, 0] = starts_local

    def _record_trajectory_points(self, root_local: torch.Tensor, done: torch.Tensor) -> None:
        stride = max(int(self.cfg.episode_route_plot_stride), 1)
        should_store = (self.episode_length_buf % stride == 0) | done
        env_ids = torch.nonzero(should_store, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        indices = self._trajectory_count[env_ids].clamp_max(self._trajectory_xyz.shape[1] - 1)
        self._trajectory_xyz[env_ids, indices] = root_local[env_ids]
        self._trajectory_count[env_ids] = (indices + 1).clamp_max(self._trajectory_xyz.shape[1])

    def _save_episode_trajectory_plot(self, env_id: int, episode_id: int) -> None:

        if not bool(self.cfg.save_episode_route_plots):
            return
        count = int(self._trajectory_count[env_id].item())
        if count < 2:
            return
        xyz = self._trajectory_xyz[env_id, :count].detach().cpu().numpy()
        route_id = int(self._route_ids[env_id].item())
        if (
            self._episode_route_plot_mode == "validation"
            and route_id in self._validation_plotted_routes
        ):
            return
        start = np.asarray(self.cfg.route_starts[route_id], dtype=np.float64)
        goal = np.asarray(self.cfg.route_goals[route_id], dtype=np.float64)
        reason_names = {1: "contact", 2: "bounds", 3: "success", 4: "timeout"}
        reason = reason_names.get(int(self._terminal_reason[env_id].item()), "unknown")
        episode_reward = float(self._episode_return[env_id].item())
        terminal_rewards = {
            "contact": float(self.cfg.contact_penalty),
            "bounds": float(self.cfg.out_of_bounds_penalty),
            "success": float(self.cfg.success_reward),
            "timeout": float(self.cfg.timeout_penalty),
        }
        terminal_reward = terminal_rewards.get(reason, 0.0)
        process_return = episode_reward - terminal_reward

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (axis_xy, axis_xz) = plt.subplots(
            1, 2, figsize=(14, 6.5), constrained_layout=True
        )
        lower, upper = self._safe_distance_bundle.spec.lower, self._safe_distance_bundle.spec.upper
        axis_xy.imshow(
            self._trajectory_obstacle_rgba_xy,
            origin="lower",
            extent=(lower[0], upper[0], lower[1], upper[1]),
            aspect="equal",
        )
        terminal_colors = {
            "contact": "#c9342f",
            "bounds": "#c9342f",
            "success": "#16834b",
            "timeout": "#d49322",
        }
        route_color = terminal_colors.get(reason, "#2367a8")
        axis_xy.plot(xyz[:, 0], xyz[:, 1], color=route_color, linewidth=1.5, zorder=3)
        axis_xy.scatter([start[0]], [start[1]], c=["#16834b"], s=38, label="start", zorder=4)
        axis_xy.scatter([goal[0]], [goal[1]], c=["#d49322"], s=38, label="goal", zorder=4)
        axis_xy.scatter(
            [xyz[-1, 0]],
            [xyz[-1, 1]],
            c=[route_color],
            edgecolors="#202020",
            s=32,
            label="end",
            zorder=5,
        )
        axis_xy.set(
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            title="XY trajectory with obstacles",
        )
        axis_xy.grid(alpha=0.2)
        axis_xy.legend(loc="upper right")

        axis_xz.plot(xyz[:, 0], xyz[:, 2], color=route_color, linewidth=1.5, zorder=3)
        axis_xz.scatter([start[0]], [start[2]], c=["#16834b"], s=38, zorder=4)
        axis_xz.scatter([goal[0]], [goal[2]], c=["#d49322"], s=38, zorder=4)
        axis_xz.scatter(
            [xyz[-1, 0]],
            [xyz[-1, 2]],
            c=[route_color],
            edgecolors="#202020",
            s=32,
            zorder=5,
        )
        axis_xz.set(
            xlabel="x (m)",
            ylabel="z (m)",
            xlim=(lower[0], upper[0]),
            ylim=(lower[2], upper[2]),
            title="XZ trajectory",
        )
        axis_xz.set_aspect("auto")
        axis_xz.grid(alpha=0.2)

        if reason == "success":
            outcome_title = f"Success Reward={episode_reward:.2f}"
        elif reason == "timeout":
            outcome_title = f"Timeout Reward={episode_reward:.2f}"
        else:
            outcome_title = f"False Reward={episode_reward:.2f}"
        episode_steps = int(self.episode_length_buf[env_id].item())
        if self._episode_route_plot_mode == "validation":
            figure.suptitle(
                f"{outcome_title} | Validation deterministic | "
                f"Checkpoint={self._validation_checkpoint_transitions:,} | Route={route_id}\n"
                f"Steps={episode_steps} | Reason={reason} | "
                f"Process={process_return:.2f} | Terminal={terminal_reward:.2f}",
                fontsize=14,
            )
        else:
            figure.suptitle(
                f"{outcome_title} | Process={process_return:.2f} | Terminal={terminal_reward:.2f}\n"
                f"EP={episode_id:09d} | Env={env_id:02d} | Route={route_id} | Reason={reason} | "
                f"Steps={episode_steps}",
                fontsize=14,
            )

        output_dir = Path(self.cfg.episode_route_plot_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._episode_route_plot_mode == "validation":
            output_path = output_dir / f"base_validation_route_{route_id}.png"
        else:
            output_path = output_dir / f"base_route_{route_id}_episode_{episode_id:09d}.png"
        figure.savefig(output_path, dpi=140)
        plt.close(figure)
        if self._episode_route_plot_mode == "validation":
            self._validation_plotted_routes.add(route_id)
        if self._episode_route_plot_mode == "training":
            files = sorted(
                output_dir.glob("base_route_*_episode_*.png"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            excess = len(files) - int(self.cfg.episode_route_plot_max_files)
            for old_path in files[: max(excess, 0)]:
                old_path.unlink(missing_ok=True)


from isaaclab.envs import DirectRLEnvCfg


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]


RAY_STATIC_CONTRACT: dict[str, object] = {
    "contract_version": "base_physx_ray_seven_route_multilevel_2026_07_31_r1",
    "routes": [0, 1, 2, 3, 4, 5, 6],
    "num_envs": 32,
    "physics_hz": 50,
    "policy_hz": 10,
    "episode_seconds": 400.0,
    "body_semantics": {"forward": "-X", "left": "-Y", "up": "+Z"},
    "action": {
        "order": ["F_forward", "F_left", "F_up", "T_pitch", "T_yaw"],
        "limits": [90.0, 60.0, 45.0, 18.0, 18.0],
        "physx_mapping": ["Fx=-90*a0", "Fy=-60*a1", "Fz=45*a2", "Ty=18*a3", "Tz=18*a4"],
        "roll_pd": {"kp": 6.0, "kd": 3.0, "limit_nm": 8.0},
    },
    "physics": {
        "gravity": [0.0, 0.0, 0.0],
        "linear_damping": 2.5,
        "angular_damping": 4.0,
        "collision_detection": "physx_gpu_discrete_contact_50hz",
        "sweep_ccd": False,
        "speculative_ccd": False,
        "solver": "TGS",
        "solver_position_iterations": 8,
        "solver_velocity_iterations": 2,
        "contact_offset_m": 0.002,
        "rest_offset_m": 0.0,
        "restitution": 0.0,
        "current_hydro_thrusters": False,
    },
    "ray_depth": {
        "frames": 4,
        "height": 45,
        "width": 80,
        "horizontal_fov_deg": 110.0,
        "vertical_fov_deg": 70.0,
        "max_distance_m": 60.0,
        "encoding": "log1p(d)/log(61)",
    },
    "state": {
        "dim": 28,
        "order": [
            "dvl_velocity_semantic_3",
            "dvl_ranges_log_4",
            "dvl_hit_flags_4",
            "imu_semantic_quaternion_wxyz_4",
            "imu_semantic_angular_rate_3",
            "pressure_depth_1",
            "goal_direction_semantic_3",
            "goal_distance_1",
            "previous_action_5",
        ],
    },
    "workspace_bounds_m": [[-135.0, 135.0], [-90.0, 90.0], [24.0, 92.0]],
    "pressure_depth_normalization": {
        "water_surface_z_m": 120.0,
        "center_m": 62.0,
        "scale_m": 34.0,
    },
    "safe_distance": {
        "resolution_m": 0.5,
        "core_clearance_m": 0.75,
        "extension_contact_clearance_m": 0.002,
        "delta_cap_m": 0.25,
    },
    "reward": {
        "safe_distance_delta_scale": 8.0,
        "time_cost": -0.05,
        "action_sq_scale": -0.01,
        "action_delta_sq_scale": -0.005,
        "contact": -300.0,
        "out_of_bounds": -300.0,
        "success": 300.0,
        "timeout": -100.0,
        "priority": ["contact", "out_of_bounds", "success", "timeout"],
    },
    "success_radius_m": 2.0,
    "collision_shapes": ["body_main", "camera_housing", "thruster_envelope"],
    "removed_reference_geometry": [
        "visual_obstacles_x10/Cylinder",
        "visual_obstacles_x10/Cylinder_001",
        "visual_obstacles_x10/Cylinder_002",
        "visual_obstacles_x10/Cylinder_003",
        "visual_obstacles_x10/Cylinder_004",
        "pipe_colliders_x10/pipe_000_Cylinder",
        "pipe_colliders_x10/pipe_001_Cylinder_001",
        "pipe_colliders_x10/pipe_002_Cylinder_002",
        "pipe_colliders_x10/pipe_003_Cylinder_003",
        "pipe_colliders_x10/pipe_004_Cylinder_004",
    ],
}
RAY_STATIC_CONTRACT_HASH = hashlib.sha256(
    json.dumps(RAY_STATIC_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@configclass
class BsaD2RayEnvCfg(DirectRLEnvCfg):

    episode_length_s = 400.0
    decimation = 5
    is_finite_horizon = False
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    observation_space = spaces.Dict(
        {
            "ray_depth": spaces.Box(low=-1.0, high=1.0, shape=(4, 45, 80), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32),
        }
    )
    state_space = 0
    debug_vis = False

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 50.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            enable_ccd=False,
            enable_stabilization=False,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=300.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    bsa_d2_usd_path = str(_REPO_ROOT / "assets" / "robot" / "bsa_d2" / "usd" / "bsa_d2_asset.usd")
    bsa_d2_reference_prim_path = "/World/BSA_D2"
    auv_prim_path = "/World/envs/env_.*/BSA_D2/base_link"
    auv_source_root_path = "/World/envs/env_0/BSA_D2"
    auv_source_body_path = "/World/envs/env_0/BSA_D2/base_link"
    auv_collision_root_relative_path = "collisions"
    expected_auv_collision_names = ("body_main", "camera_housing", "thruster_envelope")
    hide_auv_cad_visual = True

    avoid_x10_usd_path = str(
        _REPO_ROOT
        / "assets"
        / "env"
        / "oil_rig_obstacle"
        / "oil_rig_obstacle"
        / "usd"
        / "oil_rig_avoidance_scene.usd"
    )
    avoid_x10_reference_prim_path = "/oil_rig_avoidance"
    avoid_x10_source_prim_path = "/World/envs/env_0/oil_rig_avoidance"
    avoid_x10_raw_scene_bbox_min = (-18.63, -11.8615, -6.9235)
    avoid_x10_raw_scene_bbox_max = (17.4791, 11.8615, 15.9243)
    avoid_x10_scale = 200.0 / (17.4791 - (-18.63))
    avoid_x10_target_center_xy = (0.0, 0.0)
    seabed_z_m = -6.9235
    avoid_x10_removed_prim_paths = tuple(RAY_STATIC_CONTRACT["removed_reference_geometry"])
    pipe_collider_count_expected = 197
    visual_obstacle_mesh_count_expected = 9
    hide_avoid_x10_visual_obstacles = True

    workspace_bounds = ((-135.0, 135.0), (-90.0, 90.0), (24.0, 92.0))
    route_starts = (
        (-125.0, 0.00, 56.34),
        (-125.0, -32.85, 56.34),
        (-125.0, 32.85, 56.34),
        (-125.0, -32.85, 56.34),
        (-125.0, 32.85, 56.34),
        (-80.0, 55.0, 56.34),
        (-80.0, -55.0, 56.34),
    )
    route_goals = (
        (125.0, 0.00, 56.34),
        (125.0, -32.85, 56.34),
        (125.0, 32.85, 56.34),
        (125.0, 32.85, 56.34),
        (125.0, -32.85, 56.34),
        (95.0, -55.0, 56.34),
        (95.0, 55.0, 56.34),
    )
    active_route_ids = (0, 1, 2, 3, 4, 5, 6)
    goal_radius_m = 2.0

    force_limits_n = (90.0, 60.0, 45.0)
    torque_limits_nm = (18.0, 18.0)
    roll_pd_kp_nm_rad = 6.0
    roll_pd_kd_nm_s_rad = 3.0
    roll_pd_limit_nm = 8.0
    linear_damping = 2.5
    angular_damping = 4.0
    solver_position_iteration_count = 8
    solver_velocity_iteration_count = 2
    contact_offset_m = 0.002
    rest_offset_m = 0.0

    ray_depth_width = 80
    ray_depth_height = 45
    ray_depth_history = 4
    ray_depth_max_distance_m = 60.0
    ray_depth_horizontal_fov_deg = 110.0
    ray_depth_vertical_fov_deg = 70.0
    ray_camera_offset_pos_m = (-0.5218, 0.0564, 0.1577)
    ray_camera_offset_rot_wxyz = (0.0, 0.0, 0.0, 1.0)
    ray_camera_focal_length = 24.0
    ray_camera_horizontal_aperture = 2.0 * ray_camera_focal_length * math.tan(
        math.radians(ray_depth_horizontal_fov_deg / 2.0)
    )
    ray_camera_vertical_aperture = 2.0 * ray_camera_focal_length * math.tan(
        math.radians(ray_depth_vertical_fov_deg / 2.0)
    )

    dvl_offset_pos_m = (0.0, 0.0, -0.20)
    dvl_elevation_deg = 22.5
    dvl_min_range_m = 0.1
    dvl_max_range_m = 100.0
    dvl_dropout_missing_beams = 2
    dvl_velocity_scales_m_s = (2.0, 1.5, 1.0)
    imu_angular_rate_scale_rad_s = 3.0
    water_surface_z_m = 120.0
    water_density_kg_m3 = 1000.0
    pressure_gravity_m_s2 = 9.81
    atmosphere_pressure_pa = 101325.0
    pressure_depth_center_m = 62.0
    pressure_depth_scale_m = 34.0
    goal_distance_scale_m = 300.0

    safe_distance_resolution_m = 0.5
    safe_distance_core_clearance_m = 0.75
    safe_distance_extension_clearance_m = 0.002
    safe_distance_delta_cap_m = 0.25
    safe_distance_progress_scale = 8.0
    safe_distance_cache_dir = str(_REPO_ROOT / "outputs" / "base_safe_distance_cache")
    safe_distance_preflight_dir = str(_REPO_ROOT / "outputs" / "base_safe_distance_preflight")
    warp_cache_path = str(_REPO_ROOT / "outputs" / "base_warp_cache")

    time_cost = -0.05
    action_cost_scale = 0.01
    action_delta_cost_scale = 0.005
    success_reward = 300.0
    contact_penalty = -300.0
    out_of_bounds_penalty = -300.0
    timeout_penalty = -100.0

    save_episode_route_plots = True
    episode_route_plot_dir = str(_REPO_ROOT / "outputs" / "base_episode_routes")
    episode_route_plot_max_files = 100
    episode_route_plot_stride = 10
    episode_route_plot_mode = "training"
    validation_checkpoint_transitions = -1

    base_static_contract_hash = RAY_STATIC_CONTRACT_HASH


"""inspection base-compatible PhysX ray environment with uniform water current.

The base five-dimensional direct wrench, roll PD, observations, rewards,
route geometry, and hard PhysX contact termination are inherited unchanged.
inspection adds only a fixed horizontal current per episode and the corresponding
linear/quadratic relative-water-velocity drag force.
"""


from collections.abc import Sequence


from isaaclab.utils.math import quat_apply_inverse


class BsaD2InspectionEnv(BsaD2RayEnv):
    """inspection direct-wrench task with hidden, episode-fixed current."""

    cfg: BsaD2InspectionEnvCfg

    def __init__(
        self,
        cfg: BsaD2InspectionEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._validation_plotted_current_cases: set[tuple[int, int, int]] = set()

        if str(cfg.episode_route_plot_mode).strip().lower() == "training":
            pattern = re.compile(r"episode_(\d+)\.png$")
            existing_ids = []
            for path in Path(cfg.episode_route_plot_dir).glob("inspection_route_*_episode_*.png"):
                match = pattern.search(path.name)
                if match:
                    existing_ids.append(int(match.group(1)))
            self._plot_serial = max(int(self._plot_serial), max(existing_ids, default=-1) + 1)

        self._current_world = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._current_body = torch.zeros_like(self._current_world)
        self._current_force_b = torch.zeros_like(self._current_world)
        self._current_speed_m_s = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._current_direction_offset_deg = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._current_combo_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._current_max_force_n = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_max_relative_water_speed_m_s = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._fixed_route_ids: torch.Tensor | None = None
        self._fixed_current_speed_m_s: torch.Tensor | None = None
        self._fixed_current_direction_offset_deg: torch.Tensor | None = None

        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._sample_episode_current(all_env_ids)
        self.inspection_contract_hash = str(self._resume_signature["runtime_contract_hash"])
        self.inspection_geometry_hash = str(self._safe_distance_bundle.cache_id)
        self._update_current_extras()

    def _make_resume_signature(self) -> dict[str, object]:
        """Build a inspection signature while reusing base's exact geometry digest."""

        base_signature = super()._make_resume_signature()
        payload: dict[str, object] = {
            "static_contract": INSPECTION_STATIC_CONTRACT,
            "static_contract_hash": INSPECTION_STATIC_CONTRACT_HASH,
            "geometry": base_signature["geometry"],
        }
        payload["runtime_contract_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    def get_inspection_resume_signature(self, indices=None) -> dict[str, object]:
        """Return a JSON-safe inspection contract signature for launchers/checkpoints."""

        del indices
        return json.loads(json.dumps(self._resume_signature, sort_keys=True))

    def _sample_episode_current(self, env_ids: torch.Tensor) -> None:
        """Sample one of 15 route-relative current conditions per environment."""

        if env_ids.numel() == 0:
            return
        if not bool(self.cfg.current_episode_fixed):
            raise ValueError("inspection current_episode_fixed must be True")
        speeds = torch.as_tensor(
            self.cfg.current_speed_levels_m_s, dtype=torch.float32, device=self.device
        )
        offsets = torch.as_tensor(
            self.cfg.current_direction_offsets_deg,
            dtype=torch.float32,
            device=self.device,
        )
        if speeds.numel() != 3 or offsets.numel() != 5:
            raise ValueError(
                "the current grid must combine 3 speeds and 5 directions for 15 conditions"
            )
        if torch.any(speeds <= 0.0):
            raise ValueError("current speed must be positive")

        forced_speed = self._fixed_current_speed_m_s
        forced_offset = self._fixed_current_direction_offset_deg
        if forced_speed is not None and forced_offset is not None:
            sampled_speed = forced_speed[env_ids]
            sampled_offset = forced_offset[env_ids]
            speed_idx = torch.argmin(
                torch.abs(sampled_speed.unsqueeze(1) - speeds.unsqueeze(0)), dim=1
            )
            offset_idx = torch.argmin(
                torch.abs(sampled_offset.unsqueeze(1) - offsets.unsqueeze(0)), dim=1
            )
            if not torch.allclose(sampled_speed, speeds[speed_idx], atol=1.0e-6):
                raise ValueError("inspection fixed current speed is outside the contract")
            if not torch.allclose(sampled_offset, offsets[offset_idx], atol=1.0e-6):
                raise ValueError("inspection fixed current offset is outside the contract")
            combo = speed_idx * int(offsets.numel()) + offset_idx
        else:
            combo = torch.randint(
                low=0,
                high=int(speeds.numel() * offsets.numel()),
                size=(env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
            speed_idx = torch.div(combo, offsets.numel(), rounding_mode="floor")
            offset_idx = torch.remainder(combo, offsets.numel())
            sampled_speed = speeds[speed_idx]
            sampled_offset = offsets[offset_idx]

        route_starts = self._route_starts[self._route_ids[env_ids]]
        route_goals = self._route_goals[self._route_ids[env_ids]]
        route_xy = route_goals[:, :2] - route_starts[:, :2]
        route_norm = torch.linalg.norm(route_xy, dim=1, keepdim=True).clamp_min(1.0e-8)
        route_forward = route_xy / route_norm
        route_left = torch.stack((-route_forward[:, 1], route_forward[:, 0]), dim=1)
        radians = torch.deg2rad(sampled_offset)
        current_xy = sampled_speed.unsqueeze(1) * (
            torch.cos(radians).unsqueeze(1) * route_forward
            + torch.sin(radians).unsqueeze(1) * route_left
        )
        current_world = torch.cat(
            (current_xy, torch.zeros(env_ids.numel(), 1, device=self.device)), dim=1
        )

        self._current_combo_index[env_ids] = combo
        self._current_speed_m_s[env_ids] = sampled_speed
        self._current_direction_offset_deg[env_ids] = sampled_offset
        self._current_world[env_ids] = current_world
        self._current_body[env_ids] = quat_apply_inverse(
            self._auv.data.root_quat_w[env_ids], current_world
        )
        self._current_force_b[env_ids] = 0.0
        self._current_max_force_n[env_ids] = 0.0
        self._episode_max_relative_water_speed_m_s[env_ids] = 0.0

    def set_fixed_route_ids(self, route_ids: Sequence[int] | torch.Tensor) -> None:
        """Set route IDs used by the next reset (validation only)."""

        values = torch.as_tensor(route_ids, dtype=torch.long, device=self.device).flatten()
        if values.numel() != self.num_envs:
            raise ValueError(
                f"inspection fixed route count must equal num_envs={self.num_envs}, got {values.numel()}"
            )
        valid = torch.as_tensor(self.cfg.active_route_ids, dtype=torch.long, device=self.device)
        if not torch.isin(values, valid).all():
            raise ValueError("inspection fixed route IDs contain an inactive route")
        self._fixed_route_ids = values.clone()
        self._route_ids.copy_(values)
        self._transition_route_ids.copy_(values)

    def set_fixed_current_cases(
        self,
        speeds_m_s: Sequence[float] | torch.Tensor,
        offsets_deg: Sequence[float] | torch.Tensor,
    ) -> None:
        """Set one deterministic current case per environment before reset."""

        speeds = torch.as_tensor(speeds_m_s, dtype=torch.float32, device=self.device).flatten()
        offsets = torch.as_tensor(offsets_deg, dtype=torch.float32, device=self.device).flatten()
        if speeds.numel() != self.num_envs or offsets.numel() != self.num_envs:
            raise ValueError("inspection fixed current arrays must have one value per environment")
        self._fixed_current_speed_m_s = speeds.clone()
        self._fixed_current_direction_offset_deg = offsets.clone()

    def _apply_action(self) -> None:
        """Apply base wrench plus relative-water-velocity current drag."""

        super()._apply_action()
        if not hasattr(self, "_current_world"):
            return
        if not bool(self.cfg.current_enabled):
            return

        current_body = quat_apply_inverse(self._auv.data.root_quat_w, self._current_world)
        v_rel_body = self._auv.data.root_lin_vel_b - current_body
        linear = torch.as_tensor(
            self.cfg.current_linear_damping_body, dtype=torch.float32, device=self.device
        )
        quadratic = torch.as_tensor(
            self.cfg.current_quadratic_damping_body,
            dtype=torch.float32,
            device=self.device,
        )
        hydro_force = -linear.unsqueeze(0) * v_rel_body - quadratic.unsqueeze(0) * torch.abs(
            v_rel_body
        ) * v_rel_body
        if not torch.isfinite(hydro_force).all():
            raise RuntimeError("current hydrodynamics contain NaN or Inf")

        self._current_body.copy_(current_body)
        self._current_force_b.copy_(hydro_force)
        self._current_max_force_n = torch.maximum(
            self._current_max_force_n, torch.linalg.norm(hydro_force, dim=1)
        )
        self._episode_max_relative_water_speed_m_s = torch.maximum(
            self._episode_max_relative_water_speed_m_s,
            torch.linalg.norm(v_rel_body, dim=1),
        )
        self._wrench_forces_b += hydro_force.unsqueeze(1)
        self._auv.permanent_wrench_composer.set_forces_and_torques(
            forces=self._wrench_forces_b,
            torques=self._wrench_torques_b,
            body_ids=[0],
            is_global=False,
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            normalized = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.device, dtype=torch.long)
        else:
            normalized = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(normalized)
        if hasattr(self, "_current_world"):
            self._sample_episode_current(normalized)

    def _update_current_extras(self) -> None:
        """Expose diagnostics to the wrapper without adding actor features."""

        if not hasattr(self, "extras") or not hasattr(self, "_current_world"):
            return
        self.extras["inspection_current_speed_m_s"] = self._current_speed_m_s.clone()
        self.extras["inspection_current_direction_offset_deg"] = (
            self._current_direction_offset_deg.clone()
        )
        self.extras["inspection_current_world_xyz"] = self._current_world.clone()
        self.extras["inspection_current_body_xyz"] = self._current_body.clone()
        self.extras["inspection_current_force_body_n"] = self._current_force_b.clone()
        self.extras["inspection_current_max_force_n"] = self._current_max_force_n.clone()
        self.extras["inspection_current_combo_index"] = self._current_combo_index.clone()
        self.extras["inspection_current_offset_deg"] = self._current_direction_offset_deg.clone()
        self.extras["inspection_current_class_id"] = self._current_combo_index.clone()
        self.extras["inspection_episode_max_hydro_force_n"] = self._current_max_force_n.clone()
        self.extras["inspection_episode_max_relative_water_speed_m_s"] = (
            self._episode_max_relative_water_speed_m_s.clone()
        )

    @staticmethod
    def _current_label(offset_deg: float) -> str:
        return {
            0: "following",
            45: "forward_left_45",
            -45: "forward_right_45",
            90: "left_cross",
            -90: "right_cross",
        }.get(int(round(offset_deg)), f"offset_{int(round(offset_deg)):+d}")

    def _save_episode_trajectory_plot(self, env_id: int, episode_id: int) -> None:
        """Use inspection filenames and add the sampled current to the image banner."""

        mode = str(self.cfg.episode_route_plot_mode).strip().lower()
        route_id = int(self._route_ids[env_id].item())
        speed = float(self._current_speed_m_s[env_id].item())
        offset = float(self._current_direction_offset_deg[env_id].item())
        validation_case_key = (
            route_id,
            int(round(speed * 1000.0)),
            int(round(offset)),
        )
        if mode == "validation":
            if validation_case_key in self._validation_plotted_current_cases:
                return
            self._validation_plotted_routes.discard(route_id)
        super()._save_episode_trajectory_plot(env_id, episode_id)
        if not bool(self.cfg.save_episode_route_plots):
            return
        output_dir = Path(self.cfg.episode_route_plot_dir).resolve()
        label = self._current_label(offset)
        if mode == "validation":
            source = output_dir / f"base_validation_route_{route_id}.png"
            target = output_dir / (
                f"inspection_validation_route_{route_id}_current_{speed:.1f}mps_{int(round(offset)):+d}deg.png"
            )
        else:
            source = output_dir / f"base_route_{route_id}_episode_{episode_id:09d}.png"
            target = output_dir / f"inspection_route_{route_id}_episode_{episode_id:09d}.png"
        if not source.is_file():
            return
        source.replace(target)
        try:
            from PIL import Image, ImageDraw

            image = Image.open(target).convert("RGB")
            draw = ImageDraw.Draw(image)
            reason_name = {
                1: "False",
                2: "False",
                3: "Success",
                4: "Timeout",
            }.get(int(self._terminal_reason[env_id].item()), "False")
            outcome = f"{reason_name} Reward={float(self._episode_return[env_id].item()):.2f}"
            banner = (
                f"{outcome} | inspection current={speed:.1f} m/s {label} offset={offset:+.0f} deg | "
                f"maxF={float(self._current_max_force_n[env_id].item()):.2f} N"
            )
            draw.rectangle((0, 0, image.width, 32), fill=(255, 255, 255))
            draw.text((8, 8), banner, fill=(20, 20, 20))
            image.save(target)
        except Exception:
            pass
        if mode == "validation":
            self._validation_plotted_current_cases.add(validation_case_key)
        if mode == "training":
            files = sorted(
                output_dir.glob("inspection_route_*_episode_*.png"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            excess = len(files) - int(self.cfg.episode_route_plot_max_files)
            for old_path in files[: max(excess, 0)]:
                old_path.unlink(missing_ok=True)

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        self._update_current_extras()
        return terminated, truncated


"""inspection current-disturbance configuration built on the base PhysX ray task.

inspection deliberately keeps the base action, observation, reward, geometry, and
collision contracts.  The only new physical input is a per-episode, uniform
horizontal water-current disturbance applied by the inspection environment.
"""


import copy


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]


INSPECTION_STATIC_CONTRACT = copy.deepcopy(RAY_STATIC_CONTRACT)
INSPECTION_STATIC_CONTRACT["contract_version"] = (
    "inspection_physx_ray_seven_route_fixed_current_2026_07_31_r1"
)
INSPECTION_STATIC_CONTRACT["physics"]["current_hydro_thrusters"] = False
INSPECTION_STATIC_CONTRACT["physics"]["current_hydrodynamics"] = True
INSPECTION_STATIC_CONTRACT["physics"]["current_model"] = "uniform_episode_fixed_body_drag"
INSPECTION_STATIC_CONTRACT["current"] = {
    "episode_fixed": True,
    "speed_m_s": [0.1, 0.3, 0.5],
    "direction_offsets_deg": [0.0, 45.0, -45.0, 90.0, -90.0],
    "relative_to": "route_forward",
    "horizontal_only": True,
    "linear_damping_body": [8.0, 8.0, 6.0],
    "quadratic_damping_body": [4.0, 4.0, 3.0],
    "actor_observation": "hidden",
}
INSPECTION_STATIC_CONTRACT_HASH = hashlib.sha256(
    json.dumps(INSPECTION_STATIC_CONTRACT, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()


@configclass
class BsaD2InspectionEnvCfg(BsaD2RayEnvCfg):
    """inspection direct-wrench environment with fixed per-episode water current."""

    current_enabled: bool = True
    current_episode_fixed: bool = True
    current_speed_levels_m_s = (0.1, 0.3, 0.5)
    current_direction_offsets_deg = (0.0, 45.0, -45.0, 90.0, -90.0)
    current_linear_damping_body = (8.0, 8.0, 6.0)
    current_quadratic_damping_body = (4.0, 4.0, 3.0)
    current_actor_observation_enabled: bool = False

    safe_distance_cache_dir = str(_REPO_ROOT / "outputs" / "base_safe_distance_cache")
    safe_distance_preflight_dir = str(_REPO_ROOT / "outputs" / "base_safe_distance_preflight")
    warp_cache_path = str(_REPO_ROOT / "outputs" / "base_warp_cache")
    episode_route_plot_dir = str(_REPO_ROOT / "outputs" / "inspection_current_episode_routes")

    inspection_static_contract_hash: str = INSPECTION_STATIC_CONTRACT_HASH


"""strong-path base-compatible PhysX ray environment with uniform water current.

The inspection five-dimensional direct wrench, roll PD, ray stack, rewards, route
geometry, current physics, and hard PhysX contact termination are inherited
unchanged. strong-path adds only the compact privileged teacher state and versioned
training/validation metadata.
"""


class BsaD2StrongPathEnv(BsaD2InspectionEnv):
    """strong-path direct-wrench teacher task with privileged compact state."""

    cfg: BsaD2StrongPathEnvCfg

    def __init__(
        self,
        cfg: BsaD2StrongPathEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        if str(cfg.episode_route_plot_mode).strip().lower() == "training":
            pattern = re.compile(r"episode_(\d+)\.png$")
            existing_ids = []
            for path in Path(cfg.episode_route_plot_dir).glob("strongpath_route_*_episode_*.png"):
                match = pattern.search(path.name)
                if match:
                    existing_ids.append(int(match.group(1)))
            self._plot_serial = max(int(self._plot_serial), max(existing_ids, default=-1) + 1)

        self.strongpath_contract_hash = str(self._resume_signature["runtime_contract_hash"])
        self.strongpath_geometry_hash = str(self._safe_distance_bundle.cache_id)
        self._load_strongpath_strong_paths()
        self._previous_oracle_progress = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._last_oracle_progress_delta = torch.zeros_like(self._previous_oracle_progress)
        self._last_oracle_cross_track_m = torch.zeros_like(self._previous_oracle_progress)
        self._last_oracle_distance_to_next_m = torch.zeros_like(self._previous_oracle_progress)
        self._last_oracle_progress_ratio = torch.zeros_like(self._previous_oracle_progress)
        self._episode_oracle_cross_track_sum_m = torch.zeros_like(self._previous_oracle_progress)
        self._episode_oracle_cross_track_max_m = torch.zeros_like(self._previous_oracle_progress)
        self._episode_oracle_waypoint_distance_sum_m = torch.zeros_like(self._previous_oracle_progress)
        self._episode_oracle_metric_count = torch.zeros_like(self._previous_oracle_progress)
        self._update_current_extras()

    def _make_resume_signature(self) -> dict[str, object]:
        """Build a strong-path signature while reusing base's exact geometry digest."""

        base_signature = super()._make_resume_signature()
        payload: dict[str, object] = {
            "static_contract": STRONGPATH_STATIC_CONTRACT,
            "static_contract_hash": STRONGPATH_STATIC_CONTRACT_HASH,
            "geometry": base_signature["geometry"],
        }
        payload["runtime_contract_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    def get_strongpath_resume_signature(self, indices=None) -> dict[str, object]:
        """Return a JSON-safe strong-path contract signature for launchers/checkpoints."""

        del indices
        return json.loads(json.dumps(self._resume_signature, sort_keys=True))

    def _load_strongpath_strong_paths(self) -> None:

        manifest_path = Path(self.cfg.strong_path_manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"strong-path Oracle path manifest not found: {manifest_path}.run build_strongpath_strong_paths.cmd."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths: dict[int, torch.Tensor] = {}
        cumulative: dict[int, torch.Tensor] = {}
        min_clearance: dict[int, float] = {}
        max_points = 0
        for route_id in self.cfg.active_route_ids:
            route_payload = manifest.get("routes", {}).get(str(route_id))
            if not isinstance(route_payload, dict):
                raise KeyError(f"strong-path Oracle manifest missing route={route_id}")
            npz_path = Path(str(route_payload["npz"])).resolve()
            if not npz_path.is_file():
                raise FileNotFoundError(f"strong-path Oracle route={route_id} npz not found: {npz_path}")
            with np.load(npz_path) as data:
                path_np = np.asarray(data["smoothed_path"], dtype=np.float32)
            if path_np.ndim != 2 or path_np.shape[1] != 3 or path_np.shape[0] < 2:
                raise ValueError(f"strong-path Oracle route={route_id} path shape invalid: {path_np.shape}")
            path = torch.as_tensor(path_np, dtype=torch.float32, device=self.device)
            segment = path[1:] - path[:-1]
            segment_len = torch.linalg.norm(segment, dim=1).clamp_min(1.0e-6)
            cum = torch.cat(
                (torch.zeros(1, dtype=torch.float32, device=self.device), torch.cumsum(segment_len, dim=0))
            )
            paths[int(route_id)] = path
            cumulative[int(route_id)] = cum
            min_clearance[int(route_id)] = float(route_payload.get("min_clearance_m", 0.0))
            max_points = max(max_points, int(path.shape[0]))
        self._strongpath_oracle_manifest = manifest
        self._strongpath_strong_paths = paths
        self._strongpath_oracle_cumulative = cumulative
        self._strongpath_oracle_min_clearance = min_clearance
        self._strongpath_oracle_max_points = max_points

    def _project_to_strongpath_strong_path(
        self, root_local: torch.Tensor, route_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        projections = []
        tangents = []
        progresses = []
        next_waypoints = []
        cross_tracks = []
        distance_to_next = []
        min_clearance_ahead = []
        lookahead = float(self.cfg.oracle_lookahead_m)
        for env_index in range(root_local.shape[0]):
            route_id = int(route_ids[env_index].item())
            path = self._strongpath_strong_paths[route_id]
            cum = self._strongpath_oracle_cumulative[route_id]
            pos = root_local[env_index]
            start = path[:-1]
            end = path[1:]
            segment = end - start
            segment_len_sq = torch.sum(segment * segment, dim=1).clamp_min(1.0e-8)
            t = torch.sum((pos.unsqueeze(0) - start) * segment, dim=1) / segment_len_sq
            t = t.clamp(0.0, 1.0)
            projected = start + t.unsqueeze(1) * segment
            dist2 = torch.sum((projected - pos.unsqueeze(0)) ** 2, dim=1)
            best = int(torch.argmin(dist2).item())
            best_projection = projected[best]
            seg_len = torch.sqrt(segment_len_sq[best]).clamp_min(1.0e-6)
            tangent = segment[best] / seg_len
            progress = cum[best] + t[best] * seg_len
            target_progress = torch.clamp(progress + lookahead, max=cum[-1])
            waypoint_index = int(torch.searchsorted(cum, target_progress).clamp(1, path.shape[0] - 1).item())
            next_waypoint = path[waypoint_index]

            projections.append(best_projection)
            tangents.append(tangent)
            progresses.append(progress)
            next_waypoints.append(next_waypoint)
            cross_tracks.append(torch.sqrt(dist2[best]).clamp_min(0.0))
            distance_to_next.append(torch.linalg.norm(next_waypoint - pos))
            min_clearance_ahead.append(
                torch.tensor(self._strongpath_oracle_min_clearance[route_id], dtype=torch.float32, device=self.device)
            )

        return {
            "projection": torch.stack(projections, dim=0),
            "tangent": torch.stack(tangents, dim=0),
            "progress": torch.stack(progresses, dim=0),
            "next_waypoint": torch.stack(next_waypoints, dim=0),
            "cross_track": torch.stack(cross_tracks, dim=0),
            "distance_to_next": torch.stack(distance_to_next, dim=0),
            "min_clearance_ahead": torch.stack(min_clearance_ahead, dim=0),
        }

    def _semantic_body_vector(self, root_quat: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:

        physical_b = quat_apply_inverse(root_quat, vector_w)
        return torch.stack((-physical_b[:, 0], -physical_b[:, 1], physical_b[:, 2]), dim=1)

    def _strongpath_ray_direction_cache(self) -> torch.Tensor:
        """Return [H*W, 3] semantic-body ray directions for sector clearances."""

        cached = getattr(self, "_strongpath_ray_dirs_semantic", None)
        if cached is not None and cached.device == self.device:
            return cached
        height = int(self.cfg.ray_depth_height)
        width = int(self.cfg.ray_depth_width)
        h_fov = math.radians(float(self.cfg.ray_depth_horizontal_fov_deg))
        v_fov = math.radians(float(self.cfg.ray_depth_vertical_fov_deg))
        ys = torch.linspace(
            math.tan(h_fov * 0.5),
            -math.tan(h_fov * 0.5),
            width,
            dtype=torch.float32,
            device=self.device,
        )
        zs = torch.linspace(
            math.tan(v_fov * 0.5),
            -math.tan(v_fov * 0.5),
            height,
            dtype=torch.float32,
            device=self.device,
        )
        grid_z, grid_y = torch.meshgrid(zs, ys, indexing="ij")
        dirs = torch.stack((torch.ones_like(grid_y), grid_y, grid_z), dim=-1)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1.0e-8)
        self._strongpath_ray_dirs_semantic = dirs.reshape(-1, 3)
        self._strongpath_sector_masks = {
            "front": torch.ones(height * width, dtype=torch.bool, device=self.device),
            "left": self._strongpath_ray_dirs_semantic[:, 1] > 0.25,
            "right": self._strongpath_ray_dirs_semantic[:, 1] < -0.25,
            "up": self._strongpath_ray_dirs_semantic[:, 2] > 0.20,
            "down": self._strongpath_ray_dirs_semantic[:, 2] < -0.20,
        }
        return self._strongpath_ray_dirs_semantic

    def _build_state_observation(self) -> torch.Tensor:

        base_state = super()._build_state_observation()
        root_pos_w = self._auv.data.root_pos_w
        root_quat = self._auv.data.root_quat_w
        root_local = root_pos_w - self._env_origins()
        route_ids = self._route_ids
        oracle = self._project_to_strongpath_strong_path(root_local, route_ids)

        current_body_physical = getattr(self, "_current_body", torch.zeros_like(root_pos_w))
        current_body_semantic = torch.stack(
            (-current_body_physical[:, 0], -current_body_physical[:, 1], current_body_physical[:, 2]), dim=1
        )
        max_current = max(float(max(self.cfg.current_speed_levels_m_s)), 1.0e-6)
        current_body_norm = (current_body_semantic / max_current).clamp(-1.0, 1.0)

        next_waypoint_error_w = oracle["next_waypoint"] + self._env_origins() - root_pos_w
        next_waypoint_body = self._semantic_body_vector(root_quat, next_waypoint_error_w)
        next_waypoint_norm = (
            next_waypoint_body / float(self.cfg.oracle_distance_scale_m)
        ).clamp(-1.0, 1.0)

        tangent_body = self._semantic_body_vector(root_quat, oracle["tangent"])
        tangent_body = tangent_body / torch.linalg.norm(tangent_body, dim=1, keepdim=True).clamp_min(1.0e-6)
        tangent_body = tangent_body.clamp(-1.0, 1.0)

        cross_track_norm = (
            oracle["cross_track"].unsqueeze(1) / float(self.cfg.oracle_cross_track_scale_m)
        ).clamp(0.0, 1.0)
        route_total = torch.stack(
            [self._strongpath_oracle_cumulative[int(r.item())][-1] for r in route_ids], dim=0
        ).unsqueeze(1).clamp_min(1.0e-6)
        progress_norm = (oracle["progress"].unsqueeze(1) / route_total).clamp(0.0, 1.0)
        distance_to_next_norm = (
            oracle["distance_to_next"].unsqueeze(1) / float(self.cfg.oracle_distance_scale_m)
        ).clamp(0.0, 1.0)

        dsafe, dsafe_valid = self._safe_distance_runtime.lookup(root_local, route_ids)
        planned_current_clearance = torch.where(dsafe_valid, dsafe, torch.zeros_like(dsafe))
        planned_current_clearance_norm = (
            planned_current_clearance.unsqueeze(1) / float(self.cfg.oracle_clearance_scale_m)
        ).clamp(0.0, 1.0)
        planned_min_ahead_norm = (
            oracle["min_clearance_ahead"].unsqueeze(1) / float(self.cfg.oracle_clearance_scale_m)
        ).clamp(0.0, 1.0)

        self._strongpath_ray_direction_cache()
        latest_depth_m = torch.expm1(
            self._ray_depth_history[:, -1] * math.log1p(float(self.cfg.ray_depth_max_distance_m))
        ).reshape(self.num_envs, -1)
        max_ray = float(self.cfg.ray_depth_max_distance_m)
        margin_scale = max_ray
        sector_values = []
        for name in ("front", "left", "right", "up", "down"):
            mask = self._strongpath_sector_masks[name]
            sector_min = torch.amin(latest_depth_m[:, mask], dim=1)
            margin = (
                (sector_min - float(self.cfg.safe_distance_core_clearance_m)) / margin_scale
            ).clamp(-1.0, 1.0)
            sector_values.append(margin)
        sector_clearance = torch.stack(sector_values, dim=1)

        state = torch.cat(
            (
                base_state,
                current_body_norm,
                next_waypoint_norm,
                tangent_body,
                cross_track_norm,
                progress_norm,
                distance_to_next_norm,
                planned_current_clearance_norm,
                planned_min_ahead_norm,
                sector_clearance,
            ),
            dim=1,
        )
        expected_dim = int(self.cfg.teacher_state_dim)
        if state.shape != (self.num_envs, expected_dim):
            raise RuntimeError(f"strong-path teacher state dimension mismatch: {tuple(state.shape)}")
        if not torch.isfinite(state).all():
            raise RuntimeError("strong-path teacher state observation contains NaN/Inf")
        return state.to(torch.float32)

    def _sample_episode_current(self, env_ids: torch.Tensor) -> None:
        """Sample one of 15 route-relative current conditions per environment."""

        if env_ids.numel() == 0:
            return
        if not bool(self.cfg.current_episode_fixed):
            raise ValueError("strong-path current_episode_fixed must be True")
        speeds = torch.as_tensor(
            self.cfg.current_speed_levels_m_s, dtype=torch.float32, device=self.device
        )
        offsets = torch.as_tensor(
            self.cfg.current_direction_offsets_deg,
            dtype=torch.float32,
            device=self.device,
        )
        if speeds.numel() != 3 or offsets.numel() != 5:
            raise ValueError("strong-path current contract must contain 3 speeds and 5 direction offsets")
        if torch.any(speeds <= 0.0):
            raise ValueError("strong-path current speeds must be positive")

        forced_speed = self._fixed_current_speed_m_s
        forced_offset = self._fixed_current_direction_offset_deg
        if forced_speed is not None and forced_offset is not None:
            sampled_speed = forced_speed[env_ids]
            sampled_offset = forced_offset[env_ids]
            speed_idx = torch.argmin(
                torch.abs(sampled_speed.unsqueeze(1) - speeds.unsqueeze(0)), dim=1
            )
            offset_idx = torch.argmin(
                torch.abs(sampled_offset.unsqueeze(1) - offsets.unsqueeze(0)), dim=1
            )
            if not torch.allclose(sampled_speed, speeds[speed_idx], atol=1.0e-6):
                raise ValueError("strong-path fixed current speed is outside the contract")
            if not torch.allclose(sampled_offset, offsets[offset_idx], atol=1.0e-6):
                raise ValueError("strong-path fixed current offset is outside the contract")
            combo = speed_idx * int(offsets.numel()) + offset_idx
        else:
            combo = torch.randint(
                low=0,
                high=int(speeds.numel() * offsets.numel()),
                size=(env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
            speed_idx = torch.div(combo, offsets.numel(), rounding_mode="floor")
            offset_idx = torch.remainder(combo, offsets.numel())
            sampled_speed = speeds[speed_idx]
            sampled_offset = offsets[offset_idx]

        route_starts = self._route_starts[self._route_ids[env_ids]]
        route_goals = self._route_goals[self._route_ids[env_ids]]
        route_xy = route_goals[:, :2] - route_starts[:, :2]
        route_norm = torch.linalg.norm(route_xy, dim=1, keepdim=True).clamp_min(1.0e-8)
        route_forward = route_xy / route_norm
        route_left = torch.stack((-route_forward[:, 1], route_forward[:, 0]), dim=1)
        radians = torch.deg2rad(sampled_offset)
        current_xy = sampled_speed.unsqueeze(1) * (
            torch.cos(radians).unsqueeze(1) * route_forward
            + torch.sin(radians).unsqueeze(1) * route_left
        )
        current_world = torch.cat(
            (current_xy, torch.zeros(env_ids.numel(), 1, device=self.device)), dim=1
        )

        self._current_combo_index[env_ids] = combo
        self._current_speed_m_s[env_ids] = sampled_speed
        self._current_direction_offset_deg[env_ids] = sampled_offset
        self._current_world[env_ids] = current_world
        self._current_body[env_ids] = quat_apply_inverse(
            self._auv.data.root_quat_w[env_ids], current_world
        )
        self._current_force_b[env_ids] = 0.0
        self._current_max_force_n[env_ids] = 0.0
        self._episode_max_relative_water_speed_m_s[env_ids] = 0.0

    def set_fixed_route_ids(self, route_ids: Sequence[int] | torch.Tensor) -> None:
        """Set route IDs used by the next reset (validation only)."""

        values = torch.as_tensor(route_ids, dtype=torch.long, device=self.device).flatten()
        if values.numel() != self.num_envs:
            raise ValueError(
                f"strong-path fixed route count must equal num_envs={self.num_envs}, got {values.numel()}"
            )
        valid = torch.as_tensor(self.cfg.active_route_ids, dtype=torch.long, device=self.device)
        if not torch.isin(values, valid).all():
            raise ValueError("strong-path fixed route IDs contain an inactive route")
        self._fixed_route_ids = values.clone()
        self._route_ids.copy_(values)
        self._transition_route_ids.copy_(values)

    def set_fixed_current_cases(
        self,
        speeds_m_s: Sequence[float] | torch.Tensor,
        offsets_deg: Sequence[float] | torch.Tensor,
    ) -> None:
        """Set one deterministic current case per environment before reset."""

        speeds = torch.as_tensor(speeds_m_s, dtype=torch.float32, device=self.device).flatten()
        offsets = torch.as_tensor(offsets_deg, dtype=torch.float32, device=self.device).flatten()
        if speeds.numel() != self.num_envs or offsets.numel() != self.num_envs:
            raise ValueError("strong-path fixed current arrays must have one value per environment")
        self._fixed_current_speed_m_s = speeds.clone()
        self._fixed_current_direction_offset_deg = offsets.clone()

    def _apply_action(self) -> None:
        """Use inspection's single action-plus-current PhysX wrench application."""

        super()._apply_action()

    def _get_rewards(self) -> torch.Tensor:

        root_local = self._auv.data.root_pos_w - self._env_origins()
        dsafe, valid = self._safe_distance_runtime.lookup(root_local, self._route_ids)
        delta_valid = valid & self._previous_dsafe_valid
        delta = torch.where(delta_valid, self._previous_dsafe - dsafe, torch.zeros_like(dsafe))
        delta = delta.clamp(
            min=-float(self.cfg.safe_distance_delta_cap_m),
            max=float(self.cfg.safe_distance_delta_cap_m),
        )

        oracle = self._project_to_strongpath_strong_path(root_local, self._route_ids)
        progress_delta = oracle["progress"] - self._previous_oracle_progress
        progress_delta = progress_delta.clamp(-1.0, 1.0)
        cross_track = oracle["cross_track"]
        distance_to_next = oracle["distance_to_next"]

        velocity_w = self._auv.data.root_lin_vel_w
        speed = torch.linalg.norm(velocity_w, dim=1).clamp_min(1.0e-6)
        velocity_dir = velocity_w / speed.unsqueeze(1)
        tangent_alignment = torch.sum(velocity_dir * oracle["tangent"], dim=1).clamp(-1.0, 1.0)
        tangent_penalty = torch.where(
            speed > 0.05,
            1.0 - tangent_alignment,
            torch.zeros_like(tangent_alignment),
        )

        process = (
            float(self.cfg.safe_distance_progress_scale) * delta
            + float(self.cfg.oracle_progress_reward_scale) * progress_delta
            - float(self.cfg.oracle_cross_track_penalty_scale) * cross_track
            - float(self.cfg.oracle_tangent_penalty_scale) * tangent_penalty
            - float(self.cfg.oracle_waypoint_distance_penalty_scale) * distance_to_next
            + float(self.cfg.time_cost)
            - float(self.cfg.action_cost_scale) * torch.sum(torch.square(self._actions), dim=1)
            - float(self.cfg.action_delta_cost_scale)
            * torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        )
        terminal = self._terminal_reason != self._TERMINAL_NONE
        reward = torch.where(terminal, torch.zeros_like(process), process)
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_CONTACT,
            torch.full_like(reward, float(self.cfg.contact_penalty)),
            reward,
        )
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_BOUNDS,
            torch.full_like(reward, float(self.cfg.out_of_bounds_penalty)),
            reward,
        )
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_SUCCESS,
            torch.full_like(reward, float(self.cfg.success_reward)),
            reward,
        )
        reward = torch.where(
            self._terminal_reason == self._TERMINAL_TIMEOUT,
            torch.full_like(reward, float(self.cfg.timeout_penalty)),
            reward,
        )

        self._last_dsafe = dsafe.detach()
        self._last_dsafe_valid = valid.detach()
        self._last_dsafe_delta = delta.detach()
        self._last_oracle_progress_delta = progress_delta.detach()
        self._last_oracle_cross_track_m = cross_track.detach()
        self._last_oracle_distance_to_next_m = distance_to_next.detach()
        route_total = torch.stack(
            [self._strongpath_oracle_cumulative[int(r.item())][-1] for r in self._route_ids], dim=0
        ).clamp_min(1.0e-6)
        self._last_oracle_progress_ratio = (oracle["progress"] / route_total).clamp(0.0, 1.0).detach()
        nonterminal_metric = (~terminal).to(torch.float32)
        self._episode_oracle_cross_track_sum_m += cross_track.detach() * nonterminal_metric
        self._episode_oracle_cross_track_max_m = torch.maximum(
            self._episode_oracle_cross_track_max_m,
            torch.where(terminal, torch.zeros_like(cross_track), cross_track.detach()),
        )
        self._episode_oracle_waypoint_distance_sum_m += distance_to_next.detach() * nonterminal_metric
        self._episode_oracle_metric_count += nonterminal_metric
        self._previous_dsafe = dsafe.detach()
        self._previous_dsafe_valid = valid.detach()
        self._previous_oracle_progress = oracle["progress"].detach()
        self._previous_actions = self._actions.detach().clone()
        self._episode_return += reward
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            normalized = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.device, dtype=torch.long)
        else:
            normalized = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(normalized)
        root_local = self._auv.data.root_pos_w[normalized] - self._env_origins()[normalized]
        oracle = self._project_to_strongpath_strong_path(root_local, self._route_ids[normalized])
        self._previous_oracle_progress[normalized] = oracle["progress"].detach()
        self._last_oracle_progress_delta[normalized] = 0.0
        self._last_oracle_cross_track_m[normalized] = oracle["cross_track"].detach()
        self._last_oracle_distance_to_next_m[normalized] = oracle["distance_to_next"].detach()
        route_total = torch.stack(
            [self._strongpath_oracle_cumulative[int(r.item())][-1] for r in self._route_ids[normalized]], dim=0
        ).clamp_min(1.0e-6)
        self._last_oracle_progress_ratio[normalized] = (oracle["progress"] / route_total).clamp(0.0, 1.0).detach()
        self._episode_oracle_cross_track_sum_m[normalized] = 0.0
        self._episode_oracle_cross_track_max_m[normalized] = 0.0
        self._episode_oracle_waypoint_distance_sum_m[normalized] = 0.0
        self._episode_oracle_metric_count[normalized] = 0.0

    def _update_current_extras(self) -> None:
        """Expose diagnostics to the wrapper without adding actor features."""

        if not hasattr(self, "extras") or not hasattr(self, "_current_world"):
            return
        self.extras["strongpath_current_speed_m_s"] = self._current_speed_m_s.clone()
        self.extras["strongpath_current_direction_offset_deg"] = (
            self._current_direction_offset_deg.clone()
        )
        self.extras["strongpath_current_world_xyz"] = self._current_world.clone()
        self.extras["strongpath_current_body_xyz"] = self._current_body.clone()
        self.extras["strongpath_current_force_body_n"] = self._current_force_b.clone()
        self.extras["strongpath_current_max_force_n"] = self._current_max_force_n.clone()
        self.extras["strongpath_current_combo_index"] = self._current_combo_index.clone()
        if hasattr(self, "_last_oracle_progress_delta") and hasattr(self, "_last_oracle_cross_track_m"):
            self.extras["strongpath_oracle_progress_delta_m"] = self._last_oracle_progress_delta.clone()
            self.extras["strongpath_oracle_cross_track_m"] = self._last_oracle_cross_track_m.clone()
            self.extras["strongpath_oracle_distance_to_next_waypoint_m"] = (
                self._last_oracle_distance_to_next_m.clone()
            )
            count = self._episode_oracle_metric_count.clamp_min(1.0)
            self.extras["strongpath_oracle_mean_cross_track_m"] = (
                self._episode_oracle_cross_track_sum_m / count
            ).clone()
            self.extras["strongpath_oracle_max_cross_track_m"] = (
                self._episode_oracle_cross_track_max_m.clone()
            )
            self.extras["strongpath_oracle_final_progress_ratio"] = (
                self._last_oracle_progress_ratio.clone()
            )
            self.extras["strongpath_oracle_mean_distance_to_next_waypoint_m"] = (
                self._episode_oracle_waypoint_distance_sum_m / count
            ).clone()
        self.extras["strongpath_current_offset_deg"] = self._current_direction_offset_deg.clone()
        self.extras["strongpath_current_class_id"] = self._current_combo_index.clone()
        self.extras["strongpath_episode_max_hydro_force_n"] = self._current_max_force_n.clone()
        self.extras["strongpath_episode_max_relative_water_speed_m_s"] = (
            self._episode_max_relative_water_speed_m_s.clone()
        )

    @staticmethod
    def _current_label(offset_deg: float) -> str:
        return {
            0: "following",
            45: "forward_left_45",
            -45: "forward_right_45",
            90: "left_cross",
            -90: "right_cross",
        }.get(int(round(offset_deg)), f"offset_{int(round(offset_deg)):+d}")

    def _save_episode_trajectory_plot(self, env_id: int, episode_id: int) -> None:
        if not bool(self.cfg.save_episode_route_plots):
            return
        count = int(self._trajectory_count[env_id].item())
        if count < 2:
            return

        mode = str(self.cfg.episode_route_plot_mode).strip().lower()
        route_id = int(self._route_ids[env_id].item())
        speed = float(self._current_speed_m_s[env_id].item())
        offset = float(self._current_direction_offset_deg[env_id].item())
        validation_case_key = (route_id, int(round(speed * 1000.0)), int(round(offset)))
        if mode == "validation":
            if validation_case_key in self._validation_plotted_current_cases:
                return

        xyz = self._trajectory_xyz[env_id, :count].detach().cpu().numpy()
        oracle_path = self._strongpath_strong_paths[route_id].detach().cpu().numpy()
        start = np.asarray(self.cfg.route_starts[route_id], dtype=np.float64)
        goal = np.asarray(self.cfg.route_goals[route_id], dtype=np.float64)

        reason_names = {1: "contact", 2: "bounds", 3: "success", 4: "timeout"}
        reason = reason_names.get(int(self._terminal_reason[env_id].item()), "unknown")
        episode_reward = float(self._episode_return[env_id].item())
        terminal_rewards = {
            "contact": float(self.cfg.contact_penalty),
            "bounds": float(self.cfg.out_of_bounds_penalty),
            "success": float(self.cfg.success_reward),
            "timeout": float(self.cfg.timeout_penalty),
        }
        terminal_reward = terminal_rewards.get(reason, 0.0)
        process_return = episode_reward - terminal_reward
        terminal_colors = {
            "contact": "#c9342f",
            "bounds": "#c9342f",
            "success": "#16834b",
            "timeout": "#d49322",
        }
        route_color = terminal_colors.get(reason, "#2367a8")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (axis_xy, axis_xz) = plt.subplots(
            1, 2, figsize=(14, 6.5), constrained_layout=True
        )
        lower, upper = self._safe_distance_bundle.spec.lower, self._safe_distance_bundle.spec.upper

        axis_xy.imshow(
            self._trajectory_obstacle_rgba_xy,
            origin="lower",
            extent=(lower[0], upper[0], lower[1], upper[1]),
            aspect="equal",
        )
        axis_xy.plot(
            oracle_path[:, 0],
            oracle_path[:, 1],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.25,
            label="oracle path",
            zorder=2,
        )
        axis_xy.plot(xyz[:, 0], xyz[:, 1], color=route_color, linewidth=1.6, label="actual", zorder=3)
        axis_xy.scatter([start[0]], [start[1]], c=["#16834b"], s=38, label="start", zorder=4)
        axis_xy.scatter([goal[0]], [goal[1]], c=["#d49322"], s=38, label="goal", zorder=4)
        axis_xy.scatter(
            [xyz[-1, 0]],
            [xyz[-1, 1]],
            c=[route_color],
            edgecolors="#202020",
            s=34,
            label="end",
            zorder=5,
        )
        axis_xy.set(
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            title="XY: actual vs Oracle path with obstacles",
        )
        axis_xy.grid(alpha=0.2)
        axis_xy.legend(loc="upper right")

        axis_xz.plot(
            oracle_path[:, 0],
            oracle_path[:, 2],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.25,
            label="oracle path",
            zorder=2,
        )
        axis_xz.plot(xyz[:, 0], xyz[:, 2], color=route_color, linewidth=1.6, label="actual", zorder=3)
        axis_xz.scatter([start[0]], [start[2]], c=["#16834b"], s=38, zorder=4)
        axis_xz.scatter([goal[0]], [goal[2]], c=["#d49322"], s=38, zorder=4)
        axis_xz.scatter(
            [xyz[-1, 0]],
            [xyz[-1, 2]],
            c=[route_color],
            edgecolors="#202020",
            s=34,
            zorder=5,
        )
        axis_xz.set(
            xlabel="x (m)",
            ylabel="z (m)",
            xlim=(lower[0], upper[0]),
            ylim=(lower[2], upper[2]),
            title="XZ: actual vs Oracle path",
        )
        axis_xz.set_aspect("auto")
        axis_xz.grid(alpha=0.2)
        axis_xz.legend(loc="upper right")

        if reason == "success":
            outcome_title = f"Success Reward={episode_reward:.2f}"
        elif reason == "timeout":
            outcome_title = f"Timeout Reward={episode_reward:.2f}"
        else:
            outcome_title = f"False Reward={episode_reward:.2f}"

        episode_steps = int(self.episode_length_buf[env_id].item())
        label = self._current_label(offset)
        metric_count = float(self._episode_oracle_metric_count[env_id].clamp_min(1.0).item())
        mean_cross = float(self._episode_oracle_cross_track_sum_m[env_id].item()) / metric_count
        max_cross = float(self._episode_oracle_cross_track_max_m[env_id].item())
        progress_ratio = float(self._last_oracle_progress_ratio[env_id].item())
        mean_wp = float(self._episode_oracle_waypoint_distance_sum_m[env_id].item()) / metric_count
        if mode == "validation":
            figure.suptitle(
                f"{outcome_title} | strong-path deterministic validation | "
                f"Checkpoint={getattr(self, '_validation_checkpoint_transitions', 0):,} | "
                f"Route={route_id} | Current={speed:.1f}m/s {label} offset={offset:+.0f}deg\n"
                f"Steps={episode_steps} | Reason={reason} | Process={process_return:.2f} | "
                f"Terminal={terminal_reward:.2f} | meanCT={mean_cross:.2f}m maxCT={max_cross:.2f}m "
                f"progress={progress_ratio:.3f} meanWP={mean_wp:.2f}m",
                fontsize=13,
            )
        else:
            figure.suptitle(
                f"{outcome_title} | strong-path current={speed:.1f}m/s {label} offset={offset:+.0f}deg | "
                f"Process={process_return:.2f} | Terminal={terminal_reward:.2f}\n"
                f"EP={episode_id:09d} | Env={env_id:02d} | Route={route_id} | Reason={reason} | "
                f"Steps={episode_steps} | meanCT={mean_cross:.2f}m maxCT={max_cross:.2f}m "
                f"progress={progress_ratio:.3f} meanWP={mean_wp:.2f}m",
                fontsize=13,
            )

        output_dir = Path(self.cfg.episode_route_plot_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == "validation":
            target = output_dir / (
                f"strongpath_validation_route_{route_id}_current_{speed:.1f}mps_{int(round(offset)):+d}deg.png"
            )
        else:
            target = output_dir / f"strongpath_route_{route_id}_episode_{episode_id:09d}.png"
        figure.savefig(target, dpi=140)
        plt.close(figure)

        if mode == "validation":
            self._validation_plotted_current_cases.add(validation_case_key)
        if mode == "training":
            files = sorted(
                output_dir.glob("strongpath_route_*_episode_*.png"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            excess = len(files) - int(self.cfg.episode_route_plot_max_files)
            for old_path in files[: max(excess, 0)]:
                old_path.unlink(missing_ok=True)

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        self._update_current_extras()
        return terminated, truncated


"""strong-path current-disturbance configuration built on the base PhysX ray task.

strong-path deliberately keeps the base action, observation, reward, geometry, and
collision contracts.  The only new physical input is a per-episode, uniform
horizontal water-current disturbance applied by the strong-path environment.
"""


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]


STRONGPATH_STATIC_CONTRACT = copy.deepcopy(INSPECTION_STATIC_CONTRACT)
STRONGPATH_STATIC_CONTRACT["contract_version"] = (
    "strongpath_strong_path_following_seven_route_fixed_current_2026_08_02_r1"
)
STRONGPATH_STATIC_CONTRACT["current"]["actor_observation"] = "privileged"
STRONGPATH_STATIC_CONTRACT["state"] = {
    "dim": 47,
    "base_base_dim": 28,
    "strong_path_extra_dim": 19,
    "current_actor_observation": "strong_path_privileged",
    "order": [
        "base_base_state_28",
        "current_velocity_semantic_body_3",
        "next_waypoint_body_3",
        "path_tangent_body_3",
        "cross_track_error_1",
        "progress_along_strong_path_1",
        "distance_to_next_waypoint_1",
        "planned_clearance_current_1",
        "planned_min_clearance_ahead_1",
        "sector_clearance_front_left_right_up_down_5",
    ],
}
STRONGPATH_STATIC_CONTRACT_HASH = hashlib.sha256(
    json.dumps(STRONGPATH_STATIC_CONTRACT, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()


@configclass
class BsaD2StrongPathEnvCfg(BsaD2InspectionEnvCfg):
    """strong-path teacher-minimal environment with fixed per-episode water current."""

    observation_space = spaces.Dict(
        {
            "ray_depth": spaces.Box(low=-1.0, high=1.0, shape=(4, 45, 80), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(47,), dtype=np.float32),
        }
    )
    teacher_state_dim: int = 47
    current_actor_observation_enabled: bool = True

    safe_distance_cache_dir = str(_REPO_ROOT / "outputs" / "base_safe_distance_cache")
    safe_distance_preflight_dir = str(_REPO_ROOT / "outputs" / "base_safe_distance_preflight")
    warp_cache_path = str(_REPO_ROOT / "outputs" / "base_warp_cache")
    episode_route_plot_dir = str(_REPO_ROOT / "outputs" / "strongpath_strong_path_episode_routes")


    strong_path_manifest_path = str(_REPO_ROOT / "outputs" / "oracle_paths" / "manifest.json")
    oracle_waypoint_spacing_m: float = 5.0
    oracle_lookahead_m: float = 8.0
    oracle_distance_scale_m: float = 30.0
    oracle_cross_track_scale_m: float = 10.0
    oracle_clearance_scale_m: float = 10.0

    oracle_progress_reward_scale: float = 8.0
    oracle_cross_track_penalty_scale: float = 0.30
    oracle_tangent_penalty_scale: float = 0.10
    oracle_waypoint_distance_penalty_scale: float = 0.05

    strongpath_static_contract_hash: str = STRONGPATH_STATIC_CONTRACT_HASH


"""SB3 bridge for inspection, including current metadata in terminal infos."""


class AuvSb3VecEnvWrapper(Sb3VecEnvWrapper):
    """Keep Dict depth/state observations and expose inspection terminal metadata."""

    _TERMINAL_LABELS = {1: "contact", 2: "bounds", 3: "success", 4: "TIMEOUT"}
    _TERMINAL_COLORS = {
        1: Fore.LIGHTRED_EX,
        2: Fore.LIGHTRED_EX,
        3: Fore.LIGHTGREEN_EX,
        4: Fore.LIGHTYELLOW_EX,
    }

    def __init__(self, env, episode_log_path: Path, fast_variant: bool = True):
        super().__init__(env, fast_variant=fast_variant)
        self._console_transitions = 0
        self._console_episode_count = 0
        self._console_progress_interval = 10_000
        self._console_progress_origin = 0
        self._console_progress_started_at = time.monotonic()
        self._episode_route_dir = Path(self.env.unwrapped.cfg.episode_route_plot_dir).resolve()
        self._episode_logger = EpisodeCsvLogger(episode_log_path)

    def set_console_transition_offset(self, transitions: int) -> None:
        self._console_transitions = int(transitions)
        self._console_progress_origin = int(transitions)
        self._console_progress_started_at = time.monotonic()

    def _process_spaces(self) -> None:
        observation_space = self.unwrapped.single_observation_space["policy"]
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("the inspection environment must output a Dict observation")
        if set(observation_space.spaces) != {"ray_depth", "state"}:
            raise ValueError("the inspection observation may only contain ray_depth and state")
        action_space = self.unwrapped.single_action_space
        if not isinstance(action_space, spaces.Box) or action_space.shape != (5,):
            raise ValueError("the inspection action must be a Box of shape (5,)")
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def _transition_route_ids(self) -> np.ndarray:
        base_env = self.env.unwrapped
        values = getattr(base_env, "_transition_route_ids", getattr(base_env, "_route_ids", None))
        if values is None:
            raise AttributeError("the inspection environment has no transition route id")
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        result = np.asarray(values, dtype=np.int64).reshape(-1).copy()
        if result.shape != (self.num_envs,):
            raise ValueError(f"inspection route id shape mismatch: {result.shape}")
        return result

    @staticmethod
    def _copy_extra_values(infos, extras, reset_ids, keys) -> None:
        for key in keys:
            value = extras.get(key)
            if value is None:
                raise KeyError(f"inspection extras missing field: {key}")
            if hasattr(value, "detach"):
                values = value.detach().cpu().numpy()
            else:
                values = np.asarray(value)
            for raw_index in reset_ids:
                env_index = int(raw_index)
                item = values[env_index]
                infos[env_index][key] = np.asarray(item).item() if np.asarray(item).ndim == 0 else np.asarray(item).copy()

    def _process_extras(self, obs, terminated, truncated, extras, reset_ids):
        infos = super()._process_extras(obs, terminated, truncated, extras, reset_ids)
        self._copy_extra_values(
            infos,
            extras,
            reset_ids,
            ("base_terminal_reason", "base_contact_point_count", "base_episode_id"),
        )
        self._copy_extra_values(
            infos,
            extras,
            reset_ids,
            (
                "inspection_current_speed_m_s",
                "inspection_current_offset_deg",
                "inspection_current_class_id",
                "inspection_episode_max_hydro_force_n",
                "inspection_episode_max_relative_water_speed_m_s",
            ),
        )
        return infos

    @staticmethod
    def _current_label(offset_deg: float) -> str:
        return {
            0: "following",
            45: "forward_left_45",
            -45: "forward_right_45",
            90: "left_cross",
            -90: "right_cross",
        }.get(int(round(offset_deg)), f"offset_{int(round(offset_deg)):+d}")

    @staticmethod
    def _speed_token(speed_m_s: float) -> str:
        return f"{speed_m_s:.1f}".replace(".", "p")

    def _print_terminal_episode(self, env_index: int, info: dict[str, Any], terminal_reward: float) -> None:
        reason = int(info.get("base_terminal_reason", 0))
        label = self._TERMINAL_LABELS.get(reason, "unknown")
        color = self._TERMINAL_COLORS.get(reason, Fore.WHITE)
        episode = info.get("episode") or {}
        length = int(episode.get("l", 0))
        episode_return = float(episode.get("r", float("nan")))
        route_id = int(info["base_route_id"])
        contact_points = int(info.get("base_contact_point_count", 0))
        episode_id = int(info.get("base_episode_id", -1))
        if episode_id < 0:
            raise RuntimeError(f"inspection done env={env_index} missing episode ID")

        speed = float(info.get("inspection_current_speed_m_s", 0.0))
        offset = float(info.get("inspection_current_offset_deg", 0.0))
        current_label = self._current_label(offset)
        class_id = int(info.get("inspection_current_class_id", -1))
        max_force = float(info.get("inspection_episode_max_hydro_force_n", 0.0))
        max_rel_speed = float(info.get("inspection_episode_max_relative_water_speed_m_s", 0.0))
        reason_name = {1: "contact", 2: "bounds", 3: "success", 4: "timeout"}.get(reason, "unknown")
        process_return = episode_return - float(terminal_reward)
        image_filename = f"inspection_route_{route_id}_episode_{episode_id:09d}.png"
        image_path = (self._episode_route_dir / image_filename).resolve()
        if not image_path.is_file():
            raise RuntimeError(f"inspection episode={episode_id} trajectory plot not found: {image_path}")

        self._console_episode_count += 1
        self._episode_logger.append(
            {
                "episode_id": episode_id,
                "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "transitions": self._console_transitions,
                "env_id": env_index,
                "route_id": route_id,
                "current_speed_m_s": f"{speed:.6f}",
                "current_offset_deg": f"{offset:.1f}",
                "current_label": current_label,
                "current_class_id": class_id,
                "max_hydro_force_n": f"{max_force:.6f}",
                "max_relative_water_speed_m_s": f"{max_rel_speed:.6f}",
                "reason": reason_name,
                "steps": length,
                "episode_return": f"{episode_return:.6f}",
                "process_return": f"{process_return:.6f}",
                "terminal_reward": f"{float(terminal_reward):.6f}",
                "contact_points": contact_points,
                "image_filename": image_filename,
                "image_path": str(image_path),
            }
        )
        print(
            color
            + (
                f"[EPISODE {episode_id:09d}] t={self._console_transitions:,} env={env_index:02d} "
                f"route={route_id} current={speed:.1f}m/s offset={offset:+.0f}deg {current_label} class={class_id} "
                f"{label} steps={length} return={episode_return:.2f} process={process_return:.2f} "
                f"terminal={float(terminal_reward):.2f} contacts={contact_points} "
                f"maxF={max_force:.2f}N maxRel={max_rel_speed:.2f}m/s image={image_filename}"
            )
            + Style.RESET_ALL,
            flush=True,
        )

    def _print_periodic_progress(self, previous_transitions: int) -> None:
        if self._console_transitions // self._console_progress_interval <= previous_transitions // self._console_progress_interval:
            return
        elapsed = max(time.monotonic() - self._console_progress_started_at, 1.0e-6)
        completed = self._console_transitions - self._console_progress_origin
        fps = completed / elapsed
        print(
            Fore.LIGHTCYAN_EX
            + f"[PROGRESS] transitions={self._console_transitions:,} episodes_this_run={self._console_episode_count:,} fps={fps:.1f}"
            + Style.RESET_ALL,
            flush=True,
        )

    def step_wait(self):
        route_ids_before_step = self._transition_route_ids()
        observations, rewards, dones, infos = super().step_wait()
        previous_transitions = self._console_transitions
        self._console_transitions += self.num_envs
        for env_index, info in enumerate(infos):
            info["base_route_id"] = int(route_ids_before_step[env_index])
            if bool(dones[env_index]):
                info["TimeLimit.truncated"] = False
                self._print_terminal_episode(env_index, info, float(rewards[env_index]))
        self._print_periodic_progress(previous_transitions)
        return observations, rewards, dones, infos


EPISODE_LOG_FIELDS = (
    "episode_id",
    "timestamp",
    "transitions",
    "env_id",
    "route_id",
    "current_speed_m_s",
    "current_offset_deg",
    "current_label",
    "current_class_id",
    "max_hydro_force_n",
    "max_relative_water_speed_m_s",
    "reason",
    "steps",
    "episode_return",
    "process_return",
    "terminal_reward",
    "contact_points",
    "image_filename",
    "image_path",
)


class EpisodeCsvLogger:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.is_file() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                actual = tuple(next(csv.reader(stream), ()))
            if actual != EPISODE_LOG_FIELDS:
                raise RuntimeError(
                    f"inspection episode CSV header mismatch: actual={actual}, expected={EPISODE_LOG_FIELDS}"
                )
            return
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=EPISODE_LOG_FIELDS).writeheader()
            stream.flush()

    def append(self, record: Mapping[str, object]) -> None:
        missing = set(EPISODE_LOG_FIELDS) - set(record)
        extra = set(record) - set(EPISODE_LOG_FIELDS)
        if missing or extra:
            raise ValueError(f"inspection episode CSV fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=EPISODE_LOG_FIELDS).writerow(
                {field: record[field] for field in EPISODE_LOG_FIELDS}
            )
            stream.flush()
