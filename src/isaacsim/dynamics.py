"""Safe distance field and dynamics utilities used by the inspection tasks."""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


_FIELD_ALGORITHM_VERSION = "exact_triangle_full_domain"

_EXACT_SIGNED_DISTANCE_MAX_POINTS = 256


_NEIGHBOR_OFFSETS: tuple[tuple[int, int, int, float], ...] = tuple(
    (dx, dy, dz, float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)

_CANONICAL_EDGE_OFFSETS: tuple[tuple[int, int, int, float], ...] = tuple(
    (dx, dy, dz, step)
    for dx, dy, dz, step in _NEIGHBOR_OFFSETS
    if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0)
)
_EDGE_SLOT_BY_OFFSET: dict[tuple[int, int, int], tuple[int, bool]] = {}
for _edge_slot, (_dx, _dy, _dz, _) in enumerate(_CANONICAL_EDGE_OFFSETS):
    _EDGE_SLOT_BY_OFFSET[(_dx, _dy, _dz)] = (_edge_slot, True)
    _EDGE_SLOT_BY_OFFSET[(-_dx, -_dy, -_dz)] = (_edge_slot, False)

_TRANSITION_CODES = np.zeros(27, dtype=np.int8)
for _offset, (_slot, _forward) in _EDGE_SLOT_BY_OFFSET.items():
    _offset_code = (_offset[0] + 1) * 9 + (_offset[1] + 1) * 3 + (_offset[2] + 1)
    _TRANSITION_CODES[_offset_code] = (_slot + 1) if _forward else -(_slot + 1)

_INTERPOLATION_CORNERS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (0, 1)
    for dy in (0, 1)
    for dz in (0, 1)
)


@dataclass(frozen=True)
class SafeDistanceFieldSpec:

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    resolution_m: float = 0.5
    clearance_m: float = 1.5
    active_route_ids: tuple[int, ...] = (0, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("A 3D safe distance field must provide lower and upper bounds for all three axes.")
        if any(float(lo) >= float(hi) for lo, hi in zip(self.lower, self.upper)):
            raise ValueError(f"workspace bounds invalid: lower={self.lower}, upper={self.upper}")
        if not np.isclose(float(self.resolution_m), 0.5, atol=1.0e-6):
            raise ValueError("exact requires exactly 0.5 m voxel resolution")
        if not np.isclose(float(self.clearance_m), 1.5, atol=1.0e-6):
            raise ValueError("This stage requires exactly 1.5 m of safety clearance inflation.")
        if not self.active_route_ids:
            raise ValueError("At least one active route is required to build the distance field.")

    @property
    def shape(self) -> tuple[int, int, int]:

        spans = (np.asarray(self.upper) - np.asarray(self.lower)) / float(self.resolution_m)
        return tuple(int(np.ceil(value - 1.0e-8)) + 1 for value in spans)


@dataclass
class SafeDistanceFieldBundle:

    spec: SafeDistanceFieldSpec
    route_slot_by_id: dict[int, int]
    distances_m: np.ndarray
    reachable: np.ndarray
    forbidden: np.ndarray
    clearance_m: np.ndarray
    cache_id: str
    planned_lengths_m: dict[int, float]
    path_points_by_route: dict[int, np.ndarray]
    transition_masks: np.ndarray | None = None
    interpolation_domain_bits: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = (len(self.route_slot_by_id),) + self.spec.shape
        if tuple(self.distances_m.shape) != expected:
            raise ValueError(f"distance fieldshape mismatch: expected={expected}, actual={self.distances_m.shape}")
        if tuple(self.reachable.shape) != expected:
            raise ValueError("reachable mask does not match the distance field shape.")
        if tuple(self.forbidden.shape) != self.spec.shape:
            raise ValueError("forbidden mask does not match the grid shape.")
        if tuple(self.clearance_m.shape) != self.spec.shape:
            raise ValueError("clearance field does not match the grid shape.")
        if not np.isfinite(self.distances_m).all():
            raise ValueError("cached distance field contains NaN/Inf; use a finite unreachable sentinel before saving.")
        if self.transition_masks is not None:
            expected_transition_shape = (len(_CANONICAL_EDGE_OFFSETS),) + self.spec.shape
            if tuple(self.transition_masks.shape) != expected_transition_shape:
                raise ValueError(
                    "transition mask shape mismatch:"
                    f"expected={expected_transition_shape}, actual={self.transition_masks.shape}"
                )
        if self.interpolation_domain_bits is not None:
            if tuple(self.interpolation_domain_bits.shape) != self.spec.shape:
                raise ValueError("interpolation domain bits do not match the grid shape.")
            if self.interpolation_domain_bits.dtype != np.uint8:
                raise ValueError("interpolation domain bits must use a uint8 bitmap.")

    def world_to_index(self, points: np.ndarray) -> np.ndarray:

        points = np.asarray(points, dtype=np.float64)
        lower = np.asarray(self.spec.lower, dtype=np.float64)
        index = np.rint((points - lower) / float(self.spec.resolution_m)).astype(np.int64)
        return index

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:

        indices = np.asarray(indices, dtype=np.float64)
        return np.asarray(self.spec.lower, dtype=np.float64) + indices * float(self.spec.resolution_m)

    def to_torch(self, device: Any):

        import torch

        route_ids = np.full(max(self.route_slot_by_id) + 1, -1, dtype=np.int64)
        for route_id, slot in self.route_slot_by_id.items():
            route_ids[int(route_id)] = int(slot)
        domain_bits = self.interpolation_domain_bits
        if domain_bits is None:
            domain_bits = np.full(self.spec.shape, 0xFF, dtype=np.uint8)
        return TorchSafeDistanceField(
            distances_m=torch.as_tensor(self.distances_m, dtype=torch.float32, device=device),
            reachable=torch.as_tensor(self.reachable, dtype=torch.bool, device=device),
            lower=torch.tensor(self.spec.lower, dtype=torch.float32, device=device),
            resolution_m=float(self.spec.resolution_m),
            route_slot_by_id=torch.as_tensor(route_ids, dtype=torch.long, device=device),
            interpolation_domain_bits=torch.as_tensor(
                domain_bits, dtype=torch.uint8, device=device
            ),
            cache_id=self.cache_id,
        )


@dataclass
class TorchSafeDistanceField:

    distances_m: Any
    reachable: Any
    lower: Any
    resolution_m: float
    route_slot_by_id: Any
    interpolation_domain_bits: Any
    cache_id: str

    def lookup(self, positions_local_m, route_ids):

        import torch

        if positions_local_m.ndim != 2 or positions_local_m.shape[1] != 3:
            raise ValueError("distance fieldquery positionsmust be [N,3]")
        route_ids = route_ids.to(dtype=torch.long)
        valid_route = torch.logical_and(route_ids >= 0, route_ids < self.route_slot_by_id.numel())
        safe_route_ids = route_ids.clamp(0, self.route_slot_by_id.numel() - 1)
        slots = self.route_slot_by_id[safe_route_ids]
        valid_route = torch.logical_and(
            valid_route,
            torch.logical_and(slots >= 0, slots < self.distances_m.shape[0]),
        )

        shape = self.distances_m.shape[1:]
        coordinate = (positions_local_m - self.lower.unsqueeze(0)) / float(self.resolution_m)
        inside = torch.all(
            torch.logical_and(coordinate >= 0.0, coordinate <= (torch.tensor(shape, device=coordinate.device) - 1.0)),
            dim=1,
        )
        coordinate = coordinate.clamp_min(0.0)
        max_index = torch.tensor(shape, device=coordinate.device, dtype=torch.float32) - 1.0
        coordinate = torch.minimum(coordinate, max_index)
        index0 = torch.floor(coordinate).to(dtype=torch.long)
        index1 = torch.minimum(index0 + 1, torch.tensor(shape, device=coordinate.device) - 1)
        weight = coordinate - index0.to(dtype=coordinate.dtype)

        values = []
        corner_valid = []
        for dx, dy, dz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
            ix = index1[:, 0] if dx else index0[:, 0]
            iy = index1[:, 1] if dy else index0[:, 1]
            iz = index1[:, 2] if dz else index0[:, 2]
            values.append(self.distances_m[slots.clamp_min(0), ix, iy, iz])
            corner_valid.append(self.reachable[slots.clamp_min(0), ix, iy, iz])

        wx0, wy0, wz0 = 1.0 - weight[:, 0], 1.0 - weight[:, 1], 1.0 - weight[:, 2]
        weights = [
            wx0 * wy0 * wz0,
            weight[:, 0] * wy0 * wz0,
            wx0 * weight[:, 1] * wz0,
            weight[:, 0] * weight[:, 1] * wz0,
            wx0 * wy0 * weight[:, 2],
            weight[:, 0] * wy0 * weight[:, 2],
            wx0 * weight[:, 1] * weight[:, 2],
            weight[:, 0] * weight[:, 1] * weight[:, 2],
        ]
        values_tensor = torch.stack(values, dim=1)
        valid_tensor = torch.stack(corner_valid, dim=1)
        weights_tensor = torch.stack(weights, dim=1)
        weighted_valid = torch.where(valid_tensor, weights_tensor, torch.zeros_like(values_tensor))
        denominator = weighted_valid.sum(dim=1)
        distance = (torch.where(valid_tensor, values_tensor, torch.zeros_like(values_tensor)) * weighted_valid).sum(dim=1)
        distance = distance / denominator.clamp_min(1.0e-6)
        required_corner = weights_tensor > 1.0e-7
        all_corners_valid = torch.logical_or(torch.logical_not(required_corner), valid_tensor).all(dim=1)
        active_axes = weight > 1.0e-7
        active_code = (
            active_axes[:, 0].to(dtype=torch.long)
            + 2 * active_axes[:, 1].to(dtype=torch.long)
            + 4 * active_axes[:, 2].to(dtype=torch.long)
        )
        domain_bits = self.interpolation_domain_bits[
            index0[:, 0], index0[:, 1], index0[:, 2]
        ].to(dtype=torch.long)
        required_domain_bit = torch.bitwise_left_shift(
            torch.ones_like(active_code), active_code
        )
        domain_valid = torch.bitwise_and(domain_bits, required_domain_bit) != 0
        valid = torch.logical_and(
            valid_route,
            torch.logical_and(
                inside,
                torch.logical_and(all_corners_valid, domain_valid),
            ),
        )
        return torch.where(valid, distance, torch.zeros_like(distance)), valid


def _python_dijkstra(
    free_mask: np.ndarray,
    seeds: Sequence[tuple[tuple[int, int, int], float]],
    resolution_m: float,
    transition_masks: np.ndarray | None = None,
) -> np.ndarray:

    shape = free_mask.shape
    distances = np.full(shape, np.inf, dtype=np.float32)
    queue: list[tuple[float, tuple[int, int, int]]] = []
    for seed_index, seed_distance in seeds:
        if not free_mask[seed_index]:
            continue
        if float(seed_distance) < float(distances[seed_index]):
            stored_seed_distance = float(np.float32(seed_distance))
            distances[seed_index] = stored_seed_distance
            heapq.heappush(queue, (stored_seed_distance, seed_index))
    while queue:
        current, index = heapq.heappop(queue)
        if current > float(distances[index]) + 1.0e-7:
            continue
        x, y, z = index
        for dx, dy, dz, geometric_step in _NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if not (0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1] and 0 <= neighbor[2] < shape[2]):
                continue
            if not free_mask[neighbor]:
                continue
            if transition_masks is None:
                if not _transition_is_clear(free_mask, index, (dx, dy, dz)):
                    continue
            elif not _edge_transition_is_allowed(
                transition_masks,
                index,
                neighbor,
                (dx, dy, dz),
            ):
                continue
            candidate = current + float(resolution_m) * geometric_step
            if candidate < float(distances[neighbor]):
                stored_candidate = float(np.float32(candidate))
                distances[neighbor] = stored_candidate
                heapq.heappush(queue, (stored_candidate, neighbor))
    return distances


def _transition_is_clear(
    free_mask: np.ndarray,
    index: tuple[int, int, int],
    offset: tuple[int, int, int],
) -> bool:

    x, y, z = index
    dx, dy, dz = offset
    x_offsets = (0, dx) if dx != 0 else (0,)
    y_offsets = (0, dy) if dy != 0 else (0,)
    z_offsets = (0, dz) if dz != 0 else (0,)
    return all(
        bool(free_mask[x + ox, y + oy, z + oz])
        for ox in x_offsets
        for oy in y_offsets
        for oz in z_offsets
    )


def _edge_transition_is_allowed(
    transition_masks: np.ndarray,
    index: tuple[int, int, int],
    neighbor: tuple[int, int, int],
    offset: tuple[int, int, int],
) -> bool:

    slot, forward = _EDGE_SLOT_BY_OFFSET[offset]
    source = index if forward else neighbor
    return bool(transition_masks[(slot,) + source])


try:
    import numba
except ImportError:
    numba = None


if numba is not None:

    @numba.njit(cache=False)
    def _numba_dijkstra(
        free_flat,
        nx,
        ny,
        nz,
        seed_nodes,
        seed_values,
        resolution_m,
        transition_masks_flat,
        transition_codes,
        use_transition_masks,
    ):

        node_count = free_flat.size
        distances = np.full(node_count, np.inf, dtype=np.float32)
        positions = np.full(node_count, -1, dtype=np.int32)
        heap_nodes = np.empty(node_count, dtype=np.int32)
        heap_values = np.empty(node_count, dtype=np.float32)
        heap_size = 0
        for seed_i in range(seed_nodes.size):
            seed_node = seed_nodes[seed_i]
            seed_value = seed_values[seed_i]
            if not free_flat[seed_node] or seed_value >= distances[seed_node]:
                continue
            distances[seed_node] = seed_value
            heap_position = positions[seed_node]
            if heap_position < 0:
                heap_position = heap_size
                heap_size += 1
                heap_nodes[heap_position] = seed_node
                positions[seed_node] = heap_position
            heap_values[heap_position] = seed_value
            while heap_position > 0:
                parent = (heap_position - 1) // 2
                if heap_values[parent] <= heap_values[heap_position]:
                    break
                parent_node = heap_nodes[parent]
                child_node = heap_nodes[heap_position]
                heap_nodes[parent], heap_nodes[heap_position] = child_node, parent_node
                heap_values[parent], heap_values[heap_position] = heap_values[heap_position], heap_values[parent]
                positions[parent_node], positions[child_node] = heap_position, parent
                heap_position = parent
        xy_stride = ny * nz
        sqrt2 = np.sqrt(2.0)
        sqrt3 = np.sqrt(3.0)

        while heap_size > 0:
            current_node = heap_nodes[0]
            current_value = heap_values[0]
            positions[current_node] = -1
            heap_size -= 1
            if heap_size > 0:
                moved_node = heap_nodes[heap_size]
                moved_value = heap_values[heap_size]
                heap_nodes[0] = moved_node
                heap_values[0] = moved_value
                positions[moved_node] = 0
                parent = 0
                while True:
                    left = parent * 2 + 1
                    if left >= heap_size:
                        break
                    right = left + 1
                    child = left
                    if right < heap_size and heap_values[right] < heap_values[left]:
                        child = right
                    if heap_values[parent] <= heap_values[child]:
                        break
                    parent_node = heap_nodes[parent]
                    child_node = heap_nodes[child]
                    heap_nodes[parent], heap_nodes[child] = child_node, parent_node
                    heap_values[parent], heap_values[child] = heap_values[child], heap_values[parent]
                    positions[parent_node], positions[child_node] = child, parent
                    parent = child

            x = current_node // xy_stride
            remainder = current_node - x * xy_stride
            y = remainder // nz
            z = remainder - y * nz
            for dx in (-1, 0, 1):
                nx_index = x + dx
                if nx_index < 0 or nx_index >= nx:
                    continue
                for dy in (-1, 0, 1):
                    ny_index = y + dy
                    if ny_index < 0 or ny_index >= ny:
                        continue
                    for dz in (-1, 0, 1):
                        nz_index = z + dz
                        if nz_index < 0 or nz_index >= nz or (dx == 0 and dy == 0 and dz == 0):
                            continue
                        neighbor = nx_index * xy_stride + ny_index * nz + nz_index
                        if not free_flat[neighbor]:
                            continue
                        if use_transition_masks:
                            code_index = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
                            transition_code = transition_codes[code_index]
                            if transition_code > 0:
                                transition_slot = transition_code - 1
                                transition_source = current_node
                            else:
                                transition_slot = -transition_code - 1
                                transition_source = neighbor
                            if not transition_masks_flat[transition_slot, transition_source]:
                                continue
                        else:
                            transition_clear = True
                            x_choice_count = 2 if dx != 0 else 1
                            y_choice_count = 2 if dy != 0 else 1
                            z_choice_count = 2 if dz != 0 else 1
                            for x_choice in range(x_choice_count):
                                check_x = x + (dx if x_choice == 1 else 0)
                                for y_choice in range(y_choice_count):
                                    check_y = y + (dy if y_choice == 1 else 0)
                                    for z_choice in range(z_choice_count):
                                        check_z = z + (dz if z_choice == 1 else 0)
                                        check_node = check_x * xy_stride + check_y * nz + check_z
                                        if not free_flat[check_node]:
                                            transition_clear = False
                                            break
                                    if not transition_clear:
                                        break
                                if not transition_clear:
                                    break
                            if not transition_clear:
                                continue
                        squared = dx * dx + dy * dy + dz * dz
                        step = 1.0 if squared == 1 else (sqrt2 if squared == 2 else sqrt3)
                        candidate = current_value + resolution_m * step
                        if candidate >= distances[neighbor]:
                            continue
                        distances[neighbor] = candidate
                        heap_position = positions[neighbor]
                        if heap_position < 0:
                            heap_position = heap_size
                            heap_size += 1
                            heap_nodes[heap_position] = neighbor
                            heap_values[heap_position] = candidate
                            positions[neighbor] = heap_position
                        else:
                            heap_values[heap_position] = candidate
                        while heap_position > 0:
                            parent = (heap_position - 1) // 2
                            if heap_values[parent] <= heap_values[heap_position]:
                                break
                            parent_node = heap_nodes[parent]
                            child_node = heap_nodes[heap_position]
                            heap_nodes[parent], heap_nodes[heap_position] = child_node, parent_node
                            heap_values[parent], heap_values[heap_position] = heap_values[heap_position], heap_values[parent]
                            positions[parent_node], positions[child_node] = heap_position, parent
                            heap_position = parent
        return distances


def dijkstra_distance_field(
    free_mask: np.ndarray,
    goal_index: tuple[int, int, int],
    resolution_m: float,
    *,
    initial_seeds: Sequence[tuple[tuple[int, int, int], float]] | None = None,
    transition_masks: np.ndarray | None = None,
    max_python_nodes: int = 250_000,
) -> np.ndarray:

    free_mask = np.asarray(free_mask, dtype=np.bool_)
    if free_mask.ndim != 3:
        raise ValueError("free_mask must be a 3D array.")
    if transition_masks is not None:
        transition_masks = np.asarray(transition_masks, dtype=np.bool_)
        expected_shape = (len(_CANONICAL_EDGE_OFFSETS),) + free_mask.shape
        if tuple(transition_masks.shape) != expected_shape:
            raise ValueError(
                "transition_masks shape mismatch:"
                f"expected={expected_shape}, actual={transition_masks.shape}"
            )
    if not all(0 <= int(index) < size for index, size in zip(goal_index, free_mask.shape)):
        raise ValueError(f"goal_index is outside the grid: {goal_index}, shape={free_mask.shape}")
    if initial_seeds is None:
        initial_seeds = ((tuple(int(value) for value in goal_index), 0.0),)
    seeds = tuple(
        (tuple(int(value) for value in index), float(distance))
        for index, distance in initial_seeds
        if all(0 <= int(index[axis]) < free_mask.shape[axis] for axis in range(3))
    )
    if not seeds or not any(free_mask[index] for index, _ in seeds):
        return np.full(free_mask.shape, np.inf, dtype=np.float32)
    if numba is not None:
        flat = np.ascontiguousarray(free_mask.reshape(-1))
        seed_nodes = np.asarray(
            [np.ravel_multi_index(index, free_mask.shape) for index, _ in seeds],
            dtype=np.int32,
        )
        seed_values = np.asarray([distance for _, distance in seeds], dtype=np.float32)
        if transition_masks is None:
            transition_masks_flat = np.zeros((1, 1), dtype=np.bool_)
            use_transition_masks = False
        else:
            transition_masks_flat = np.ascontiguousarray(
                transition_masks.reshape(len(_CANONICAL_EDGE_OFFSETS), -1)
            )
            use_transition_masks = True
        result = _numba_dijkstra(
            flat,
            int(free_mask.shape[0]),
            int(free_mask.shape[1]),
            int(free_mask.shape[2]),
            seed_nodes,
            seed_values,
            float(resolution_m),
            transition_masks_flat,
            _TRANSITION_CODES,
            use_transition_masks,
        )
        return result.reshape(free_mask.shape)
    if free_mask.size > max_python_nodes:
        raise RuntimeError(
            "building a large distance field requires numba; this Python environment has no numba, refusing to fall back to the slow implementation."
        )
    return _python_dijkstra(
        free_mask,
        seeds,
        resolution_m,
        transition_masks=transition_masks,
    )


def build_forbidden_mask(
    raw_occupied: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    resolution_m: float,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:

    from scipy import ndimage

    occupied = np.asarray(raw_occupied, dtype=np.bool_)
    if occupied.ndim != 3:
        raise ValueError("raw_occupied must be a 3D array.")
    shape = occupied.shape
    coordinates = [
        np.asarray(lower[axis], dtype=np.float64) + np.arange(shape[axis], dtype=np.float64) * float(resolution_m)
        for axis in range(3)
    ]
    if occupied.any():
        obstacle_clearance = ndimage.distance_transform_edt(
            ~occupied, sampling=float(resolution_m)
        ).astype(np.float32)
    else:
        obstacle_clearance = np.full(shape, np.inf, dtype=np.float32)
    axis_boundary_clearances = []
    for axis in range(3):
        one_dimensional = np.minimum(
            coordinates[axis] - float(lower[axis]),
            float(upper[axis]) - coordinates[axis],
        )
        reshape = tuple(shape[axis] if index == axis else 1 for index in range(3))
        axis_boundary_clearances.append(one_dimensional.reshape(reshape))
    boundary_clearance = np.minimum(
        np.minimum(axis_boundary_clearances[0], axis_boundary_clearances[1]),
        axis_boundary_clearances[2],
    ).astype(np.float32)
    clearance = np.minimum(obstacle_clearance, boundary_clearance)
    forbidden = np.logical_or(occupied, clearance < float(clearance_m) - 1.0e-5)
    return forbidden, clearance


def _point_corner_seeds(
    point: Sequence[float],
    spec: SafeDistanceFieldSpec,
    free_mask: np.ndarray,
    *,
    interpolation_domain_bits: np.ndarray | None = None,
    exact_meshes: Sequence[Any] | None = None,
) -> tuple[tuple[tuple[int, int, int], float], ...]:

    coordinate = (
        np.asarray(point, dtype=np.float64) - np.asarray(spec.lower, dtype=np.float64)
    ) / float(spec.resolution_m)
    lower_index = np.floor(coordinate).astype(np.int64)
    upper_index = np.ceil(coordinate).astype(np.int64)
    corner_indices = []
    for ix in sorted({int(lower_index[0]), int(upper_index[0])}):
        for iy in sorted({int(lower_index[1]), int(upper_index[1])}):
            for iz in sorted({int(lower_index[2]), int(upper_index[2])}):
                index = (ix, iy, iz)
                if not all(0 <= index[axis] < spec.shape[axis] for axis in range(3)):
                    return ()
                corner_indices.append(index)
    if not all(bool(free_mask[index]) for index in corner_indices):
        return ()
    if interpolation_domain_bits is not None:
        point_valid = _trilinear_valid_numpy(
            np.asarray(free_mask, dtype=np.bool_),
            np.asarray(point, dtype=np.float64).reshape(1, 3),
            spec,
            interpolation_domain_bits=interpolation_domain_bits,
        )
        if not bool(point_valid[0]):
            return ()
    seeds = []
    corner_world_points = []
    for index in corner_indices:
        world = np.asarray(spec.lower, dtype=np.float64) + np.asarray(index) * float(spec.resolution_m)
        corner_world_points.append(world)
        seeds.append((index, float(np.linalg.norm(world - np.asarray(point, dtype=np.float64)))))
    if exact_meshes is not None:
        point_array = np.asarray(point, dtype=np.float64).reshape(1, 3)
        corner_world = np.asarray(corner_world_points, dtype=np.float64)
        starts = np.repeat(point_array, len(corner_world), axis=0)
        certificate_cap = (
            float(spec.clearance_m)
            + 0.5 * float(np.sqrt(3.0)) * float(spec.resolution_m)
            + 1.0e-4
        )
        endpoint_clearance = _exact_obstacle_clearance_capped(
            np.concatenate((point_array, corner_world), axis=0),
            exact_meshes,
            certificate_cap,
        )
        segment_valid = _certify_edge_batch(
            starts,
            corner_world,
            np.repeat(endpoint_clearance[0], len(corner_world)),
            endpoint_clearance[1:],
            exact_meshes,
            required_clearance_m=float(spec.clearance_m),
            clearance_cap_m=certificate_cap,
            max_subdivision_depth=10,
        )
        if not bool(np.all(segment_valid)):
            return ()
    return tuple(seeds)


def _nearest_index(point: Sequence[float], spec: SafeDistanceFieldSpec) -> tuple[int, int, int]:
    index = np.rint(
        (np.asarray(point, dtype=np.float64) - np.asarray(spec.lower, dtype=np.float64))
        / float(spec.resolution_m)
    ).astype(np.int64)
    return tuple(int(value) for value in index)


def _trilinear_numpy(field: np.ndarray, points: np.ndarray, spec: SafeDistanceFieldSpec) -> np.ndarray:

    points = np.asarray(points, dtype=np.float64)
    coordinate = (points - np.asarray(spec.lower, dtype=np.float64)) / float(spec.resolution_m)
    inside = np.all(
        np.logical_and(coordinate >= 0.0, coordinate <= np.asarray(spec.shape, dtype=np.float64) - 1.0),
        axis=1,
    )
    clipped = np.clip(coordinate, 0.0, np.asarray(spec.shape, dtype=np.float64) - 1.0)
    index0 = np.floor(clipped).astype(np.int64)
    index1 = np.minimum(index0 + 1, np.asarray(spec.shape, dtype=np.int64) - 1)
    weight = clipped - index0
    result = np.zeros(len(points), dtype=np.float64)
    for dx, dy, dz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
        ix = index1[:, 0] if dx else index0[:, 0]
        iy = index1[:, 1] if dy else index0[:, 1]
        iz = index1[:, 2] if dz else index0[:, 2]
        corner_weight = (
            (weight[:, 0] if dx else 1.0 - weight[:, 0])
            * (weight[:, 1] if dy else 1.0 - weight[:, 1])
            * (weight[:, 2] if dz else 1.0 - weight[:, 2])
        )
        result += field[ix, iy, iz] * corner_weight
    result[~inside] = np.nan
    return result


def _trilinear_valid_numpy(
    reachable: np.ndarray,
    points: np.ndarray,
    spec: SafeDistanceFieldSpec,
    interpolation_domain_bits: np.ndarray | None = None,
) -> np.ndarray:

    points = np.asarray(points, dtype=np.float64)
    coordinate = (points - np.asarray(spec.lower, dtype=np.float64)) / float(spec.resolution_m)
    inside = np.all(
        np.logical_and(
            coordinate >= 0.0,
            coordinate <= np.asarray(spec.shape, dtype=np.float64) - 1.0,
        ),
        axis=1,
    )
    clipped = np.clip(coordinate, 0.0, np.asarray(spec.shape, dtype=np.float64) - 1.0)
    index0 = np.floor(clipped).astype(np.int64)
    index1 = np.minimum(index0 + 1, np.asarray(spec.shape, dtype=np.int64) - 1)
    weight = clipped - index0
    valid = inside.copy()
    for dx, dy, dz in (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ):
        ix = index1[:, 0] if dx else index0[:, 0]
        iy = index1[:, 1] if dy else index0[:, 1]
        iz = index1[:, 2] if dz else index0[:, 2]
        corner_weight = (
            (weight[:, 0] if dx else 1.0 - weight[:, 0])
            * (weight[:, 1] if dy else 1.0 - weight[:, 1])
            * (weight[:, 2] if dz else 1.0 - weight[:, 2])
        )
        required = corner_weight > 1.0e-7
        valid &= np.logical_or(~required, reachable[ix, iy, iz])
    if interpolation_domain_bits is not None:
        domain_bits = np.asarray(interpolation_domain_bits, dtype=np.uint8)
        if tuple(domain_bits.shape) != spec.shape:
            raise ValueError("interpolation_domain_bits and spec.shape does not match")
        active_axes = weight > 1.0e-7
        active_code = (
            active_axes[:, 0].astype(np.uint8)
            + 2 * active_axes[:, 1].astype(np.uint8)
            + 4 * active_axes[:, 2].astype(np.uint8)
        )
        selected_bits = domain_bits[index0[:, 0], index0[:, 1], index0[:, 2]]
        required_bits = np.left_shift(np.uint8(1), active_code)
        valid &= np.bitwise_and(selected_bits, required_bits) != 0
    return valid


def _densify_path_points(path_points: np.ndarray, maximum_step_m: float) -> np.ndarray:

    samples = [path_points[0]]
    for start, end in zip(path_points[:-1], path_points[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(int(np.ceil(length / maximum_step_m)), 1)
        for fraction in np.linspace(1.0 / count, 1.0, count):
            samples.append(start + fraction * (end - start))
    return np.asarray(samples)


def _bounded_signed_distance(query: Any, points: np.ndarray, *, error_context: str) -> np.ndarray:

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("bounded signed-distance query pointsmust be [N,3]")
    signed_distance = np.empty(len(points), dtype=np.float64)
    for chunk_start in range(0, len(points), _EXACT_SIGNED_DISTANCE_MAX_POINTS):
        chunk_end = min(chunk_start + _EXACT_SIGNED_DISTANCE_MAX_POINTS, len(points))
        chunk_result = np.asarray(
            query.signed_distance(points[chunk_start:chunk_end]),
            dtype=np.float64,
        )
        if not np.isfinite(chunk_result).all():
            raise RuntimeError(f"{error_context} contains NaN or Inf")
        signed_distance[chunk_start:chunk_end] = chunk_result
    return signed_distance


def _exact_mesh_clearance(
    points: np.ndarray,
    meshes: Sequence[Any],
    spec: SafeDistanceFieldSpec,
) -> np.ndarray:

    import trimesh

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("exact mesh clearance query pointsmust be [N,3]")
    lower = np.asarray(spec.lower, dtype=np.float64)
    upper = np.asarray(spec.upper, dtype=np.float64)
    boundary_clearance = np.min(
        np.concatenate((points - lower.reshape(1, 3), upper.reshape(1, 3) - points), axis=1),
        axis=1,
    )
    clearance = boundary_clearance.astype(np.float64, copy=True)
    for mesh in meshes:
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        below = np.maximum(bounds[0].reshape(1, 3) - points, 0.0)
        above = np.maximum(points - bounds[1].reshape(1, 3), 0.0)
        aabb_lower_bound = np.linalg.norm(below + above, axis=1)
        candidates = aabb_lower_bound < clearance + 1.0e-9
        if not np.any(candidates):
            continue
        query = trimesh.proximity.ProximityQuery(mesh)
        signed_distance = _bounded_signed_distance(
            query,
            points[candidates],
            error_context="exact mesh pathclearancequery",
        )
        mesh_clearance = np.where(signed_distance >= 0.0, 0.0, -signed_distance)
        clearance[candidates] = np.minimum(clearance[candidates], mesh_clearance)
    return clearance.astype(np.float32)


def _sample_path_clearance(
    path_points: np.ndarray,
    bundle: SafeDistanceFieldBundle,
    exact_meshes: Sequence[Any] | None = None,
) -> np.ndarray:

    maximum_step = float(bundle.spec.resolution_m) * 0.5
    sample_points = _densify_path_points(path_points, maximum_step)
    if exact_meshes is not None:
        return _exact_mesh_clearance(sample_points, exact_meshes, bundle.spec)
    return _trilinear_numpy(bundle.clearance_m, sample_points, bundle.spec).astype(np.float32)


def _certify_polyline_clearance(
    path_points: np.ndarray,
    meshes: Sequence[Any],
    spec: SafeDistanceFieldSpec,
    *,
    required_clearance_m: float,
    max_subdivision_depth: int = 10,
) -> float:

    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("continuous clearance certification requires at least two 3D path points.")
    if max_subdivision_depth < 0:
        raise ValueError("max_subdivision_depth must be non-negative")
    numeric_tolerance = 1.0e-5
    endpoint_clearance = _exact_mesh_clearance(points, meshes, spec).astype(np.float64)
    if float(np.min(endpoint_clearance)) < float(required_clearance_m) - numeric_tolerance:
        index = int(np.argmin(endpoint_clearance))
        raise RuntimeError(
            "exact path node exact clearance insufficient: "
            f"point={points[index].tolist()}, clearance={endpoint_clearance[index]}"
        )

    starts = points[:-1]
    ends = points[1:]
    start_clearance = endpoint_clearance[:-1]
    end_clearance = endpoint_clearance[1:]
    certified_lower_bounds: list[np.ndarray] = []
    for depth in range(max_subdivision_depth + 1):
        lengths = np.linalg.norm(ends - starts, axis=1)
        lower_bounds = np.minimum(start_clearance, end_clearance) - 0.5 * lengths
        certified = lower_bounds >= float(required_clearance_m) - numeric_tolerance
        if np.any(certified):
            certified_lower_bounds.append(lower_bounds[certified])
        unresolved = ~certified
        if not np.any(unresolved):
            return float(np.min(np.concatenate(certified_lower_bounds)))
        if depth == max_subdivision_depth:
            worst = int(np.argmin(lower_bounds[unresolved]))
            unresolved_starts = starts[unresolved]
            unresolved_ends = ends[unresolved]
            unresolved_bounds = lower_bounds[unresolved]
            raise RuntimeError(
                "exact clearance along the path cannot be certified within the maximum subdivision depth:"
                f"start={unresolved_starts[worst].tolist()}, "
                f"end={unresolved_ends[worst].tolist()}, "
                f"lower_bound={unresolved_bounds[worst]}"
            )

        unresolved_starts = starts[unresolved]
        unresolved_ends = ends[unresolved]
        unresolved_start_clearance = start_clearance[unresolved]
        unresolved_end_clearance = end_clearance[unresolved]
        midpoints = 0.5 * (unresolved_starts + unresolved_ends)
        midpoint_clearance = _exact_mesh_clearance(midpoints, meshes, spec).astype(np.float64)
        if float(np.min(midpoint_clearance)) < float(required_clearance_m) - numeric_tolerance:
            index = int(np.argmin(midpoint_clearance))
            raise RuntimeError(
                "path-segment interior exact clearance is insufficient:"
                f"point={midpoints[index].tolist()}, clearance={midpoint_clearance[index]}"
            )
        starts = np.concatenate((unresolved_starts, midpoints), axis=0)
        ends = np.concatenate((midpoints, unresolved_ends), axis=0)
        start_clearance = np.concatenate(
            (unresolved_start_clearance, midpoint_clearance),
            axis=0,
        )
        end_clearance = np.concatenate(
            (midpoint_clearance, unresolved_end_clearance),
            axis=0,
        )
    raise AssertionError("unreachable clearance certification state")


def _finite_sentinel(distance: np.ndarray, free_mask: np.ndarray, spec: SafeDistanceFieldSpec) -> tuple[np.ndarray, np.ndarray]:

    reachable = np.isfinite(distance) & free_mask
    finite_values = distance[reachable]
    if finite_values.size == 0:
        sentinel = float(np.linalg.norm(np.asarray(spec.upper) - np.asarray(spec.lower)) * 2.0)
    else:
        sentinel = float(np.max(finite_values) + np.linalg.norm(np.asarray(spec.upper) - np.asarray(spec.lower)))
    sanitized = np.where(np.isfinite(distance), distance, sentinel).astype(np.float32)
    return sanitized, reachable


def reconstruct_shortest_path(
    distance_m: np.ndarray,
    free_mask: np.ndarray,
    start_index: tuple[int, int, int],
    goal_index: tuple[int, int, int],
    resolution_m: float,
    *,
    terminal_indices: Sequence[tuple[int, int, int]] | None = None,
    transition_masks: np.ndarray | None = None,
    max_steps: int | None = None,
) -> np.ndarray:

    if not np.isfinite(distance_m[start_index]) or not free_mask[start_index]:
        raise ValueError(f"start unreachable: {start_index}")
    if max_steps is None:
        max_steps = int(np.prod(distance_m.shape) + 1)
    terminal_set = {tuple(int(value) for value in index) for index in (terminal_indices or (goal_index,))}
    current = start_index
    path = [current]
    for _ in range(max_steps):
        if current in terminal_set:
            return np.asarray(path, dtype=np.int64)
        x, y, z = current
        candidates: list[tuple[float, float, tuple[int, int, int]]] = []
        for dx, dy, dz, geometric_step in _NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if not (0 <= neighbor[0] < distance_m.shape[0] and 0 <= neighbor[1] < distance_m.shape[1] and 0 <= neighbor[2] < distance_m.shape[2]):
                continue
            if not free_mask[neighbor] or not np.isfinite(distance_m[neighbor]):
                continue
            if transition_masks is None:
                if not _transition_is_clear(free_mask, current, (dx, dy, dz)):
                    continue
            elif not _edge_transition_is_allowed(
                transition_masks,
                current,
                neighbor,
                (dx, dy, dz),
            ):
                continue
            predecessor_value = float(distance_m[neighbor]) + float(resolution_m) * geometric_step
            candidates.append((predecessor_value, float(distance_m[neighbor]), neighbor))
        if not candidates:
            raise ValueError(f"cannot backtrack from {current} to goal={goal_index} along the potential field.")
        predecessor_value, next_distance, next_index = min(candidates, key=lambda item: item[0])
        tolerance = max(float(resolution_m) * 1.0e-2, 1.0e-4)
        if predecessor_value > float(distance_m[current]) + tolerance:
            raise ValueError(
                "Dijkstra potential field is missing a predecessor satisfying the Bellman equality:"
                f"current={current}, value={float(distance_m[current])}, best={predecessor_value}"
            )
        if next_distance >= float(distance_m[current]) - 1.0e-5:
            raise ValueError(f"Dijkstra potential field is not strictly decreasing: current={current}, next={next_index}")
        current = next_index
        path.append(current)
    raise ValueError("shortest-path backtracking exceeded max_steps; the distance field is probably corrupted.")


def make_cache_id(scene_path: str | Path, scene_config: dict[str, Any], mesh_digest: str = "") -> str:

    path = Path(scene_path)
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    if path.exists():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(scene_config, sort_keys=True, default=str).encode("utf-8"))
    digest.update(str(mesh_digest).encode("ascii"))
    return digest.hexdigest()[:24]


def _mesh_digest(meshes: Iterable[Any]) -> str:

    digest = hashlib.sha256()
    for mesh in meshes:
        digest.update(np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32)).tobytes())
        digest.update(np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int32)).tobytes())
    return digest.hexdigest()


def extract_exact_pipe_meshes(
    stage,
    root_path: str,
    env_origin: Sequence[float] = (0.0, 0.0, 0.0),
):

    import isaaclab.sim as sim_utils
    from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
    from isaaclab.utils.math import matrix_from_quat
    import torch

    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim is None or not root_prim.IsValid():
        raise RuntimeError(f"exact mesh extraction failed: not found {root_path}")
    mesh_types = tuple(PRIMITIVE_MESH_TYPES) + ("Mesh",)
    mesh_prims = sim_utils.get_all_matching_child_prims(
        root_prim.GetPath(), lambda prim: prim.GetTypeName() in mesh_types
    )
    if not mesh_prims:
        raise RuntimeError(f"exact mesh extraction failed: {root_path} contains no Mesh/primitive geometry")
    root_pos, root_quat = sim_utils.resolve_prim_pose(root_prim)
    root_rotation = matrix_from_quat(torch.tensor(root_quat, dtype=torch.float32)).cpu().numpy()
    root_transform = np.eye(4, dtype=np.float64)
    root_transform[:3, :3] = root_rotation
    root_transform[:3, 3] = (
        np.asarray(root_pos, dtype=np.float64) - np.asarray(env_origin, dtype=np.float64)
    )
    meshes = []
    for prim in mesh_prims:
        if not prim.IsValid():
            raise RuntimeError(f"exact mesh extraction failed: invalid prim {prim}")
        if prim.GetTypeName() == "Mesh":
            mesh = create_trimesh_from_geom_mesh(prim)
        else:
            mesh = create_trimesh_from_geom_shape(prim)
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError(f"exact mesh extraction failed: {prim.GetPath()} has no valid triangle mesh")
        mesh.apply_scale(sim_utils.resolve_prim_scale(prim))
        relative_pos, relative_quat = sim_utils.resolve_prim_pose(prim, root_prim)
        rotation = matrix_from_quat(torch.tensor(relative_quat, dtype=torch.float32)).cpu().numpy()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(relative_pos, dtype=np.float64)
        mesh.apply_transform(transform)
        mesh.apply_transform(root_transform)
        if not bool(mesh.is_watertight) or not bool(mesh.is_volume):
            raise RuntimeError(
                f"collider {prim.GetPath()} must be a watertight positive volume with consistent normals;"
                "refusing patching, surface extraction, or bbox approximation."
            )
        meshes.append(mesh)
    return meshes


def build_exact_mesh_forbidden_mask(
    meshes: Iterable[Any],
    spec: SafeDistanceFieldSpec,
    *,
    query_chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    import trimesh

    mesh_list = tuple(meshes)
    if not mesh_list:
        raise RuntimeError("exact mesh list is empty; refusing to continue building.")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer.")
    shape = spec.shape
    shape_array = np.asarray(shape, dtype=np.int64)
    lower = np.asarray(spec.lower, dtype=np.float64)
    upper = np.asarray(spec.upper, dtype=np.float64)
    resolution = float(spec.resolution_m)
    certificate_cap = (
        float(spec.clearance_m)
        + 0.5 * float(np.sqrt(3.0)) * resolution
        + 1.0e-4
    )
    obstacle_clearance = np.full(shape, certificate_cap, dtype=np.float32)
    queried_node_count = 0

    for mesh_index, mesh in enumerate(mesh_list):
        if not bool(mesh.is_watertight) or not bool(mesh.is_volume):
            raise RuntimeError(
                f"exact collider #{mesh_index} is not a watertight positive volume, signed-distance inside/outside semantics are unreliable"
            )
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise RuntimeError(f"exact collider #{mesh_index} bounds invalid: {bounds}")
        expanded_lower = bounds[0] - certificate_cap
        expanded_upper = bounds[1] + certificate_cap
        index_lower = np.floor((expanded_lower - lower) / resolution).astype(np.int64)
        index_upper = np.ceil((expanded_upper - lower) / resolution).astype(np.int64)
        index_lower = np.maximum(index_lower, 0)
        index_upper = np.minimum(index_upper, shape_array - 1)
        if np.any(index_lower > index_upper):
            continue

        local_shape = index_upper - index_lower + 1
        local_count = int(np.prod(local_shape, dtype=np.int64))
        local_clearance = np.full(local_count, np.inf, dtype=np.float32)
        yz_stride = int(local_shape[1] * local_shape[2])
        z_size = int(local_shape[2])
        query = trimesh.proximity.ProximityQuery(mesh)
        for chunk_start in range(0, local_count, int(query_chunk_size)):
            chunk_end = min(chunk_start + int(query_chunk_size), local_count)
            flat = np.arange(chunk_start, chunk_end, dtype=np.int64)
            local_x = flat // yz_stride
            remainder = flat - local_x * yz_stride
            local_y = remainder // z_size
            local_z = remainder - local_y * z_size
            global_indices = np.stack(
                (
                    local_x + index_lower[0],
                    local_y + index_lower[1],
                    local_z + index_lower[2],
                ),
                axis=1,
            )
            points = lower.reshape(1, 3) + global_indices.astype(np.float64) * resolution
            signed_distance = _bounded_signed_distance(
                query,
                points,
                error_context=f"exact collider #{mesh_index} exact signed-distance",
            )
            local_clearance[chunk_start:chunk_end] = np.where(
                signed_distance >= 0.0,
                0.0,
                -signed_distance,
            ).astype(np.float32)
        slices = tuple(
            slice(int(index_lower[axis]), int(index_upper[axis]) + 1) for axis in range(3)
        )
        obstacle_clearance[slices] = np.minimum(
            obstacle_clearance[slices],
            local_clearance.reshape(tuple(int(value) for value in local_shape)),
        )
        queried_node_count += local_count

    if queried_node_count == 0:
        raise RuntimeError("exact collider has no valid overlap with the workspace; refusing to continue building.")

    coordinates = [
        lower[axis] + np.arange(shape[axis], dtype=np.float64) * resolution
        for axis in range(3)
    ]
    axis_boundary_clearances = []
    for axis in range(3):
        one_dimensional = np.minimum(
            coordinates[axis] - lower[axis],
            upper[axis] - coordinates[axis],
        )
        reshape = tuple(shape[axis] if index == axis else 1 for index in range(3))
        axis_boundary_clearances.append(one_dimensional.reshape(reshape))
    boundary_clearance = np.minimum(
        np.minimum(axis_boundary_clearances[0], axis_boundary_clearances[1]),
        axis_boundary_clearances[2],
    ).astype(np.float32)
    clearance = np.minimum(obstacle_clearance, boundary_clearance)
    forbidden = clearance < float(spec.clearance_m) - 1.0e-5
    return forbidden, clearance, obstacle_clearance


def _exact_obstacle_clearance_capped(
    points: np.ndarray,
    meshes: Sequence[Any],
    cap_m: float,
) -> np.ndarray:

    import trimesh

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("edge exact-clearance query pointsmust be [N,3]")
    if not np.isfinite(cap_m) or float(cap_m) <= 0.0:
        raise ValueError("edge exact-clearance cap must be a finite positive number.")
    clearance = np.full(points.shape[0], float(cap_m), dtype=np.float64)
    for mesh_index, mesh in enumerate(meshes):
        if not bool(mesh.is_watertight) or not bool(mesh.is_volume):
            raise RuntimeError(
                f"collider #{mesh_index} is not a watertight positive volume; cannot certify continuous edges."
            )
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        below = np.maximum(bounds[0].reshape(1, 3) - points, 0.0)
        above = np.maximum(points - bounds[1].reshape(1, 3), 0.0)
        aabb_lower_bound = np.linalg.norm(below + above, axis=1)
        candidates = aabb_lower_bound < clearance + 1.0e-9
        if not np.any(candidates):
            continue
        query = trimesh.proximity.ProximityQuery(mesh)
        signed_distance = _bounded_signed_distance(
            query,
            points[candidates],
            error_context=f"exact collider #{mesh_index} edge exact signed-distance",
        )
        mesh_clearance = np.where(signed_distance >= 0.0, 0.0, -signed_distance)
        clearance[candidates] = np.minimum(clearance[candidates], mesh_clearance)
    return clearance.astype(np.float32)


def _certify_edge_batch(
    starts: np.ndarray,
    ends: np.ndarray,
    start_clearance_m: np.ndarray,
    end_clearance_m: np.ndarray,
    meshes: Sequence[Any],
    *,
    required_clearance_m: float,
    clearance_cap_m: float,
    max_subdivision_depth: int = 10,
) -> np.ndarray:

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    start_clearance = np.asarray(start_clearance_m, dtype=np.float64).reshape(-1)
    end_clearance = np.asarray(end_clearance_m, dtype=np.float64).reshape(-1)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("edge batch starts/ends must both be [N,3].")
    if len(start_clearance) != len(starts) or len(end_clearance) != len(starts):
        raise ValueError("edge batch endpoint clearance length does not match the edge count.")
    if max_subdivision_depth < 0:
        raise ValueError("max_subdivision_depth must be non-negative")

    edge_count = len(starts)
    certified_result = np.zeros(edge_count, dtype=np.bool_)
    if edge_count == 0:
        return certified_result
    numeric_tolerance = 1.0e-5
    active = np.logical_and(
        start_clearance >= float(required_clearance_m) - numeric_tolerance,
        end_clearance >= float(required_clearance_m) - numeric_tolerance,
    )
    edge_ids = np.flatnonzero(active).astype(np.int64)
    certified_result[edge_ids] = True
    starts = starts[active]
    ends = ends[active]
    start_clearance = start_clearance[active]
    end_clearance = end_clearance[active]

    for depth in range(max_subdivision_depth + 1):
        if len(edge_ids) == 0:
            break
        lengths = np.linalg.norm(ends - starts, axis=1)
        lower_bounds = 0.5 * (start_clearance + end_clearance - lengths)
        certified = lower_bounds >= float(required_clearance_m) - numeric_tolerance
        unresolved = ~certified
        if not np.any(unresolved):
            break
        if depth == max_subdivision_depth:
            certified_result[edge_ids[unresolved]] = False
            break

        unresolved_ids = edge_ids[unresolved]
        unresolved_starts = starts[unresolved]
        unresolved_ends = ends[unresolved]
        unresolved_start_clearance = start_clearance[unresolved]
        unresolved_end_clearance = end_clearance[unresolved]
        midpoints = 0.5 * (unresolved_starts + unresolved_ends)
        midpoint_clearance = _exact_obstacle_clearance_capped(
            midpoints,
            meshes,
            float(clearance_cap_m),
        ).astype(np.float64)
        midpoint_safe = midpoint_clearance >= float(required_clearance_m) - numeric_tolerance
        certified_result[unresolved_ids[~midpoint_safe]] = False
        midpoint_safe &= certified_result[unresolved_ids]
        if not np.any(midpoint_safe):
            break

        kept_ids = unresolved_ids[midpoint_safe]
        kept_starts = unresolved_starts[midpoint_safe]
        kept_ends = unresolved_ends[midpoint_safe]
        kept_start_clearance = unresolved_start_clearance[midpoint_safe]
        kept_end_clearance = unresolved_end_clearance[midpoint_safe]
        kept_midpoints = midpoints[midpoint_safe]
        kept_midpoint_clearance = midpoint_clearance[midpoint_safe]
        edge_ids = np.concatenate((kept_ids, kept_ids), axis=0)
        starts = np.concatenate((kept_starts, kept_midpoints), axis=0)
        ends = np.concatenate((kept_midpoints, kept_ends), axis=0)
        start_clearance = np.concatenate(
            (kept_start_clearance, kept_midpoint_clearance),
            axis=0,
        )
        end_clearance = np.concatenate(
            (kept_midpoint_clearance, kept_end_clearance),
            axis=0,
        )
    return certified_result


def _certify_axis_aligned_domains_batch(
    lower_points: np.ndarray,
    upper_points: np.ndarray,
    meshes: Sequence[Any],
    *,
    required_clearance_m: float,
    clearance_cap_m: float,
    max_subdivision_depth: int = 4,
) -> np.ndarray:

    lower = np.asarray(lower_points, dtype=np.float64)
    upper = np.asarray(upper_points, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[1] != 3:
        raise ValueError("axis-aligned domain lower/upper must all be [N,3]")
    if max_subdivision_depth < 0:
        raise ValueError("axis-aligned domain max_subdivision_depth must be non-negative")
    result = np.ones(len(lower), dtype=np.bool_)
    if len(lower) == 0:
        return result
    original_ids = np.arange(len(lower), dtype=np.int64)
    numeric_tolerance = 1.0e-5

    for depth in range(max_subdivision_depth + 1):
        if len(original_ids) == 0:
            break
        still_possible = result[original_ids]
        if not np.any(still_possible):
            break
        lower = lower[still_possible]
        upper = upper[still_possible]
        original_ids = original_ids[still_possible]
        centers = 0.5 * (lower + upper)
        radii = 0.5 * np.linalg.norm(upper - lower, axis=1)
        center_clearance = _exact_obstacle_clearance_capped(
            centers,
            meshes,
            float(clearance_cap_m),
        ).astype(np.float64)
        center_safe = center_clearance >= float(required_clearance_m) - numeric_tolerance
        result[original_ids[~center_safe]] = False
        certified = np.logical_and(
            center_safe,
            center_clearance - radii
            >= float(required_clearance_m) - numeric_tolerance,
        )
        unresolved = np.logical_and(
            np.logical_and(center_safe, ~certified),
            result[original_ids],
        )
        if not np.any(unresolved):
            break
        if depth == max_subdivision_depth:
            result[original_ids[unresolved]] = False
            break

        unresolved_ids = original_ids[unresolved]
        unresolved_lower = lower[unresolved]
        unresolved_upper = upper[unresolved]
        unresolved_centers = centers[unresolved]
        active_axes = (unresolved_upper - unresolved_lower) > 1.0e-12
        child_lowers = []
        child_uppers = []
        child_ids = []
        for child_code in range(8):
            choose_upper = np.asarray(
                (bool(child_code & 1), bool(child_code & 2), bool(child_code & 4)),
                dtype=np.bool_,
            ).reshape(1, 3)
            keep = np.all(np.logical_or(active_axes, ~choose_upper), axis=1)
            if not np.any(keep):
                continue
            child_lowers.append(
                np.where(
                    choose_upper,
                    unresolved_centers[keep],
                    unresolved_lower[keep],
                )
            )
            child_uppers.append(
                np.where(
                    choose_upper,
                    unresolved_upper[keep],
                    unresolved_centers[keep],
                )
            )
            child_ids.append(unresolved_ids[keep])
        lower = np.concatenate(child_lowers, axis=0)
        upper = np.concatenate(child_uppers, axis=0)
        original_ids = np.concatenate(child_ids, axis=0)
    return result


def build_exact_transition_masks(
    free_mask: np.ndarray,
    obstacle_clearance_m: np.ndarray,
    meshes: Sequence[Any],
    spec: SafeDistanceFieldSpec,
    *,
    certification_chunk_size: int = 2_000,
) -> np.ndarray:

    free_mask = np.asarray(free_mask, dtype=np.bool_)
    obstacle_clearance = np.asarray(obstacle_clearance_m, dtype=np.float32)
    if tuple(free_mask.shape) != spec.shape:
        raise ValueError(f"free_mask and spec.shape does not match: {free_mask.shape} != {spec.shape}")
    if tuple(obstacle_clearance.shape) != spec.shape:
        raise ValueError("obstacle_clearance_m and spec.shape does not match")
    if not np.isfinite(obstacle_clearance).all():
        raise ValueError("obstacle_clearance_m must all be finite")
    if certification_chunk_size <= 0:
        raise ValueError("certification_chunk_size must be a positive integer.")

    shape = np.asarray(spec.shape, dtype=np.int64)
    lower_world = np.asarray(spec.lower, dtype=np.float64)
    resolution = float(spec.resolution_m)
    required_clearance = float(spec.clearance_m)
    clearance_cap = float(np.max(obstacle_clearance))
    transition_masks = np.zeros(
        (len(_CANONICAL_EDGE_OFFSETS),) + spec.shape,
        dtype=np.bool_,
    )

    for slot, (dx, dy, dz, geometric_step) in enumerate(_CANONICAL_EDGE_OFFSETS):
        offset = np.asarray((dx, dy, dz), dtype=np.int64)
        source_lower = np.maximum(0, -offset)
        source_upper = np.minimum(shape, shape - offset)
        source_slices = tuple(
            slice(int(source_lower[axis]), int(source_upper[axis])) for axis in range(3)
        )
        target_slices = tuple(
            slice(
                int(source_lower[axis] + offset[axis]),
                int(source_upper[axis] + offset[axis]),
            )
            for axis in range(3)
        )

        local_shape = tuple(int(value) for value in (source_upper - source_lower))
        support_free = np.ones(local_shape, dtype=np.bool_)
        x_offsets = (0, dx) if dx != 0 else (0,)
        y_offsets = (0, dy) if dy != 0 else (0,)
        z_offsets = (0, dz) if dz != 0 else (0,)
        for ox in x_offsets:
            for oy in y_offsets:
                for oz in z_offsets:
                    corner_offset = (ox, oy, oz)
                    corner_slices = tuple(
                        slice(
                            int(source_lower[axis] + corner_offset[axis]),
                            int(source_upper[axis] + corner_offset[axis]),
                        )
                        for axis in range(3)
                    )
                    support_free &= free_mask[corner_slices]

        start_clearance = obstacle_clearance[source_slices]
        end_clearance = obstacle_clearance[target_slices]
        edge_length = resolution * float(geometric_step)
        endpoint_lower_bound = 0.5 * (start_clearance + end_clearance - edge_length)
        local_valid = np.logical_and(
            support_free,
            endpoint_lower_bound >= required_clearance - 1.0e-5,
        )
        unresolved_coordinates = np.argwhere(
            np.logical_and(support_free, ~local_valid)
        )

        for chunk_start in range(0, len(unresolved_coordinates), int(certification_chunk_size)):
            chunk_end = min(
                chunk_start + int(certification_chunk_size),
                len(unresolved_coordinates),
            )
            local_indices = unresolved_coordinates[chunk_start:chunk_end]
            source_indices = local_indices + source_lower.reshape(1, 3)
            target_indices = source_indices + offset.reshape(1, 3)
            source_points = lower_world.reshape(1, 3) + source_indices * resolution
            target_points = lower_world.reshape(1, 3) + target_indices * resolution
            index_tuple = tuple(source_indices[:, axis] for axis in range(3))
            target_index_tuple = tuple(target_indices[:, axis] for axis in range(3))
            chunk_valid = _certify_edge_batch(
                source_points,
                target_points,
                obstacle_clearance[index_tuple],
                obstacle_clearance[target_index_tuple],
                meshes,
                required_clearance_m=required_clearance,
                clearance_cap_m=clearance_cap,
                max_subdivision_depth=4,
            )
            local_index_tuple = tuple(local_indices[:, axis] for axis in range(3))
            local_valid[local_index_tuple] = chunk_valid

        transition_masks[(slot,) + source_slices] = local_valid
    return transition_masks


def build_interpolation_domain_bits(
    free_mask: np.ndarray,
    transition_masks: np.ndarray,
    *,
    obstacle_clearance_m: np.ndarray | None = None,
    meshes: Sequence[Any] | None = None,
    spec: SafeDistanceFieldSpec | None = None,
    certification_chunk_size: int = 500,
    max_subdivision_depth: int = 4,
) -> np.ndarray:

    free_mask = np.asarray(free_mask, dtype=np.bool_)
    transition_masks = np.asarray(transition_masks, dtype=np.bool_)
    expected_transition_shape = (len(_CANONICAL_EDGE_OFFSETS),) + free_mask.shape
    if free_mask.ndim != 3:
        raise ValueError("interpolation domain free_mask must be a 3D array.")
    if tuple(transition_masks.shape) != expected_transition_shape:
        raise ValueError(
            "interpolation domain transition mask shape mismatch:"
            f"expected={expected_transition_shape}, actual={transition_masks.shape}"
        )
    exact_arguments = (obstacle_clearance_m, meshes, spec)
    exact_enabled = all(value is not None for value in exact_arguments)
    if any(value is not None for value in exact_arguments) and not exact_enabled:
        raise ValueError(
            "exact interpolation domain must provide obstacle_clearance_m, meshes and spec"
        )
    if certification_chunk_size <= 0:
        raise ValueError("interpolation domain certification_chunk_size must be a positive integer.")
    if max_subdivision_depth < 0:
        raise ValueError("interpolation domain max_subdivision_depth must be non-negative")
    if exact_enabled:
        obstacle_clearance = np.asarray(obstacle_clearance_m, dtype=np.float32)
        if tuple(obstacle_clearance.shape) != free_mask.shape:
            raise ValueError("interpolation domain obstacle clearance shape mismatch")
        if tuple(spec.shape) != free_mask.shape:
            raise ValueError("interpolation domain spec.shape and free_mask does not match")
        if not np.isfinite(obstacle_clearance).all():
            raise ValueError("interpolation domain obstacle clearance must all be finite")
        mesh_list = tuple(meshes)
        if not mesh_list:
            raise ValueError("interpolation domain exact mesh list must not be empty.")

    shape = np.asarray(free_mask.shape, dtype=np.int64)
    domain_bits = np.zeros(free_mask.shape, dtype=np.uint8)
    domain_bits[free_mask] |= np.uint8(1)
    for active_code in range(1, 8):
        active_axes = np.asarray(
            (
                bool(active_code & 1),
                bool(active_code & 2),
                bool(active_code & 4),
            ),
            dtype=np.int64,
        )
        base_shape = shape - active_axes
        base_slices = tuple(slice(0, int(base_shape[axis])) for axis in range(3))
        domain_valid = np.ones(tuple(int(value) for value in base_shape), dtype=np.bool_)
        corners = tuple(
            corner
            for corner in _INTERPOLATION_CORNERS
            if all(corner[axis] == 0 or active_axes[axis] for axis in range(3))
        )
        for first_index, first in enumerate(corners):
            for second in corners[first_index + 1 :]:
                offset = tuple(second[axis] - first[axis] for axis in range(3))
                slot, forward = _EDGE_SLOT_BY_OFFSET[offset]
                if not forward:
                    raise AssertionError(f"non-canonical interpolation corner pair: {first}->{second}")
                edge_slices = tuple(
                    slice(int(first[axis]), int(first[axis] + base_shape[axis]))
                    for axis in range(3)
                )
                domain_valid &= transition_masks[(slot,) + edge_slices]
        if exact_enabled and int(np.sum(active_axes)) >= 2:
            corner_clearance = []
            for corner in corners:
                corner_slices = tuple(
                    slice(int(corner[axis]), int(corner[axis] + base_shape[axis]))
                    for axis in range(3)
                )
                corner_clearance.append(obstacle_clearance[corner_slices])
            minimum_corner_clearance = np.minimum.reduce(corner_clearance)
            domain_radius = (
                0.5
                * float(spec.resolution_m)
                * float(np.sqrt(np.sum(active_axes)))
            )
            exact_valid = np.logical_and(
                domain_valid,
                minimum_corner_clearance - domain_radius
                >= float(spec.clearance_m) - 1.0e-5,
            )
            unresolved = np.argwhere(np.logical_and(domain_valid, ~exact_valid))
            lower_world = np.asarray(spec.lower, dtype=np.float64)
            active_extent = active_axes.astype(np.float64) * float(spec.resolution_m)
            clearance_cap = float(np.max(obstacle_clearance))
            for chunk_start in range(0, len(unresolved), int(certification_chunk_size)):
                chunk_end = min(
                    chunk_start + int(certification_chunk_size),
                    len(unresolved),
                )
                local_indices = unresolved[chunk_start:chunk_end]
                domain_lower = lower_world.reshape(1, 3) + (
                    local_indices.astype(np.float64) * float(spec.resolution_m)
                )
                domain_upper = domain_lower + active_extent.reshape(1, 3)
                certified = _certify_axis_aligned_domains_batch(
                    domain_lower,
                    domain_upper,
                    mesh_list,
                    required_clearance_m=float(spec.clearance_m),
                    clearance_cap_m=clearance_cap,
                    max_subdivision_depth=max_subdivision_depth,
                )
                local_tuple = tuple(local_indices[:, axis] for axis in range(3))
                exact_valid[local_tuple] = certified
            domain_valid = exact_valid
        domain_bits[base_slices] |= (
            domain_valid.astype(np.uint8) * np.uint8(1 << active_code)
        )
    return domain_bits


def constrain_transitions_to_interpolation_domains(
    transition_masks: np.ndarray,
    interpolation_domain_bits: np.ndarray,
) -> np.ndarray:

    transition_masks = np.asarray(transition_masks, dtype=np.bool_).copy()
    domain_bits = np.asarray(interpolation_domain_bits, dtype=np.uint8)
    if transition_masks.ndim != 4 or transition_masks.shape[0] != len(_CANONICAL_EDGE_OFFSETS):
        raise ValueError("constrain transition_masks shape mismatch")
    if tuple(domain_bits.shape) != tuple(transition_masks.shape[1:]):
        raise ValueError("constrain interpolation_domain_bits shape mismatch")

    shape = np.asarray(domain_bits.shape, dtype=np.int64)
    for slot, (dx, dy, dz, _) in enumerate(_CANONICAL_EDGE_OFFSETS):
        offset = np.asarray((dx, dy, dz), dtype=np.int64)
        source_lower = np.maximum(0, -offset)
        source_upper = np.minimum(shape, shape - offset)
        source_slices = tuple(
            slice(int(source_lower[axis]), int(source_upper[axis])) for axis in range(3)
        )
        domain_lower = source_lower + np.minimum(offset, 0)
        domain_upper = source_upper + np.minimum(offset, 0)
        domain_slices = tuple(
            slice(int(domain_lower[axis]), int(domain_upper[axis])) for axis in range(3)
        )
        active_code = (
            (1 if dx != 0 else 0)
            + (2 if dy != 0 else 0)
            + (4 if dz != 0 else 0)
        )
        required_bit = np.uint8(1 << active_code)
        transition_masks[(slot,) + source_slices] &= (
            np.bitwise_and(domain_bits[domain_slices], required_bit) != 0
        )
    return transition_masks


def preflight_route_fields(
    bundle: SafeDistanceFieldBundle,
    starts_by_route: dict[int, Sequence[float]],
    goals_by_route: dict[int, Sequence[float]],
    *,
    diagnostics_dir: str | Path | None = None,
    exact_meshes: Sequence[Any] | None = None,
) -> dict[int, dict[str, float | int | bool]]:

    summaries: dict[int, dict[str, float | int | bool]] = {}
    active_routes = set(bundle.spec.active_route_ids)
    if set(starts_by_route) != active_routes or set(goals_by_route) != active_routes:
        raise RuntimeError(
            "preflight route set is incomplete:"
            f"active={sorted(active_routes)}, starts={sorted(starts_by_route)}, goals={sorted(goals_by_route)}"
        )
    if diagnostics_dir is not None:
        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)
    euclidean_by_route = {}
    for route_id in bundle.spec.active_route_ids:
        start = starts_by_route[route_id]
        goal = goals_by_route[route_id]
        slot = bundle.route_slot_by_id[route_id]
        free_mask = ~bundle.forbidden
        start_seeds = _point_corner_seeds(
            start,
            bundle.spec,
            free_mask,
            interpolation_domain_bits=bundle.interpolation_domain_bits,
            exact_meshes=exact_meshes,
        )
        goal_seeds = _point_corner_seeds(
            goal,
            bundle.spec,
            free_mask,
            interpolation_domain_bits=bundle.interpolation_domain_bits,
            exact_meshes=exact_meshes,
        )
        if not start_seeds:
            raise RuntimeError(f"route {route_id} exact start has nofree interpolation corner: {start}")
        if not goal_seeds:
            raise RuntimeError(f"route {route_id} exact goal has nofree interpolation corner: {goal}")
        reachable_starts = [
            (index, offset, float(bundle.distances_m[(slot,) + index]) + offset)
            for index, offset in start_seeds
            if bundle.reachable[(slot,) + index]
        ]
        if not reachable_starts:
            raise RuntimeError(f"route {route_id} start cannot reach goal.")
        start_index, start_offset, planned_length = min(reachable_starts, key=lambda item: item[2])
        goal_seed_indices = tuple(index for index, _ in goal_seeds)
        goal_index = min(goal_seeds, key=lambda item: item[1])[0]
        path_indices = reconstruct_shortest_path(
            bundle.distances_m[slot],
            bundle.reachable[slot],
            start_index,
            goal_index,
            bundle.spec.resolution_m,
            terminal_indices=goal_seed_indices,
            transition_masks=bundle.transition_masks,
        )
        path_points = np.concatenate(
            (
                np.asarray(start, dtype=np.float64).reshape(1, 3),
                bundle.index_to_world(path_indices),
                np.asarray(goal, dtype=np.float64).reshape(1, 3),
            ),
            axis=0,
        )
        sampled_path_points = _densify_path_points(
            path_points,
            float(bundle.spec.resolution_m) * 0.5,
        )
        runtime_valid = _trilinear_valid_numpy(
            bundle.reachable[slot],
            sampled_path_points,
            bundle.spec,
            interpolation_domain_bits=bundle.interpolation_domain_bits,
        )
        if not bool(np.all(runtime_valid)):
            first_invalid = sampled_path_points[int(np.flatnonzero(~runtime_valid)[0])]
            raise RuntimeError(
                f"route {route_id} backtracked path violates the runtime trilinear valid domain: point={first_invalid.tolist()}"
            )
        path_clearance = _sample_path_clearance(
            path_points,
            bundle,
            exact_meshes=exact_meshes,
        )
        euclidean = float(np.linalg.norm(np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)))
        reconstructed_length = float(np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))
        if not np.isfinite(planned_length) or planned_length + 1.0e-4 < euclidean:
            raise RuntimeError(
                f"route {route_id} minimum safe length is invalid: planned={planned_length}, euclidean={euclidean}"
            )
        if abs(reconstructed_length - planned_length) > 0.05:
            raise RuntimeError(
                f"route {route_id} backtracked length does not match the potential field:"
                f"planned={planned_length}, reconstructed={reconstructed_length}"
            )
        if not np.isfinite(path_clearance).all():
            raise RuntimeError(f"route {route_id} pathclearancequerycontains NaN or Inf")
        if float(np.min(path_clearance)) + 1.0e-5 < float(bundle.spec.clearance_m):
            raise RuntimeError(f"route {route_id} backtrackingpathclearanceinsufficient: min={float(np.min(path_clearance))}")
        certified_clearance_lower_bound = float(np.min(path_clearance))
        if exact_meshes is not None:
            certified_clearance_lower_bound = _certify_polyline_clearance(
                path_points,
                exact_meshes,
                bundle.spec,
                required_clearance_m=float(bundle.spec.clearance_m),
            )
        bundle.planned_lengths_m[route_id] = planned_length
        bundle.path_points_by_route[route_id] = path_points
        euclidean_by_route[route_id] = euclidean
        summaries[route_id] = {
            "reachable": True,
            "euclidean_length_m": euclidean,
            "planned_length_m": planned_length,
            "reconstructed_length_m": reconstructed_length,
            "length_ratio": planned_length / max(euclidean, 1.0e-6),
            "path_voxels": int(len(path_indices)),
            "min_clearance_m": float(np.min(path_clearance)),
            "certified_clearance_lower_bound_m": certified_clearance_lower_bound,
        }
        if diagnostics_dir is not None:
            np.savez_compressed(
                Path(diagnostics_dir) / f"route{route_id}_shortest_path.npz",
                xyz=path_points.astype(np.float32),
                clearance_m=path_clearance.astype(np.float32),
            )
            _save_route_diagnostic_figure(
                route_id,
                path_points,
                start,
                goal,
                Path(diagnostics_dir) / f"route{route_id}_xy_xz_z_profile.png",
            )
    r3_r4_difference = abs(
        float(bundle.planned_lengths_m.get(3, 0.0)) - float(bundle.planned_lengths_m.get(4, 0.0))
    )
    route0_topology = _classify_route0_topology(
        bundle.path_points_by_route.get(0, np.empty((0, 3), dtype=np.float64)),
        starts_by_route.get(0, (0.0, 0.0, 0.0)),
    )
    continuity = _audit_interpolation_continuity(bundle)
    if diagnostics_dir is not None:
        (Path(diagnostics_dir) / "preflight_summary.json").write_text(
            json.dumps(
                {
                    "cache_id": bundle.cache_id,
                    "routes": summaries,
                    "route3_route4_length_difference_m": r3_r4_difference,
                    "route0_selected_topology": route0_topology,
                    "interpolation_continuity": continuity,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return summaries


def _save_route_diagnostic_figure(
    route_id: int,
    path_points: np.ndarray,
    start: Sequence[float],
    goal: Sequence[float],
    output_path: Path,
) -> None:

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distance_along = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(path_points[:, 0], path_points[:, 1], color="#1261a0", linewidth=1.4)
    axes[0].scatter([start[0], goal[0]], [start[1], goal[1]], c=["#18864b", "#d43f3a"], s=24)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title=f"Route {route_id} XY", aspect="equal")
    axes[1].plot(path_points[:, 0], path_points[:, 2], color="#7b3f98", linewidth=1.4)
    axes[1].scatter([start[0], goal[0]], [start[2], goal[2]], c=["#18864b", "#d43f3a"], s=24)
    axes[1].set(xlabel="x (m)", ylabel="z (m)", title="XZ")
    axes[2].plot(distance_along, path_points[:, 2], color="#1b7f79", linewidth=1.4)
    axes[2].set(xlabel="path distance (m)", ylabel="z (m)", title="z profile")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _classify_route0_topology(path_points: np.ndarray, start: Sequence[float]) -> str:

    if path_points.size == 0:
        return "unavailable"
    y_offset = path_points[:, 1] - float(start[1])
    z_offset = path_points[:, 2] - float(start[2])
    largest = max(float(np.max(np.abs(y_offset))), float(np.max(np.abs(z_offset))))
    if largest < 1.0:
        return "central_gap_or_straight"
    candidates = {
        "left": float(np.max(y_offset)),
        "right": float(-np.min(y_offset)),
        "over": float(np.max(z_offset)),
        "under": float(-np.min(z_offset)),
    }
    return max(candidates, key=candidates.get)


def _audit_interpolation_continuity(bundle: SafeDistanceFieldBundle) -> dict[str, float | int | bool]:

    if not np.isfinite(bundle.distances_m).all():
        raise RuntimeError("exact interpolation preflight detected NaN/Inf")
    maximum_neighbor_jump = 0.0
    checked_edges = 0
    for slot in range(bundle.distances_m.shape[0]):
        field = bundle.distances_m[slot]
        valid = bundle.reachable[slot]
        for axis in range(3):
            left_slice = [slice(None), slice(None), slice(None)]
            right_slice = [slice(None), slice(None), slice(None)]
            left_slice[axis] = slice(0, -1)
            right_slice[axis] = slice(1, None)
            valid_pairs = valid[tuple(left_slice)] & valid[tuple(right_slice)]
            if bundle.transition_masks is not None:
                axis_offset = tuple(1 if index == axis else 0 for index in range(3))
                edge_slot, forward = _EDGE_SLOT_BY_OFFSET[axis_offset]
                if not forward:
                    raise AssertionError(f"axis offset is not canonical: {axis_offset}")
                valid_pairs &= bundle.transition_masks[(edge_slot,) + tuple(left_slice)]
            if valid_pairs.any():
                jumps = np.abs(field[tuple(left_slice)] - field[tuple(right_slice)])[valid_pairs]
                maximum_neighbor_jump = max(maximum_neighbor_jump, float(np.max(jumps)))
                checked_edges += int(jumps.size)
    continuous = checked_edges > 0 and maximum_neighbor_jump <= float(bundle.spec.resolution_m) + 1.0e-3
    if not continuous:
        raise RuntimeError(
            "interpolation continuity check failed:"
            f"edges={checked_edges}, max_jump={maximum_neighbor_jump:.6f}"
        )
    return {
        "finite": True,
        "checked_neighbor_edges": checked_edges,
        "max_axis_neighbor_jump_m": maximum_neighbor_jump,
        "continuous": True,
    }


def build_exact_safe_distance_field(
    *,
    stage,
    collider_root_path: str,
    scene_path: str | Path,
    spec: SafeDistanceFieldSpec,
    scene_config: dict[str, Any],
    cache_dir: str | Path,
    starts_by_route: dict[int, Sequence[float]],
    goals_by_route: dict[int, Sequence[float]],
    env_origin: Sequence[float] = (0.0, 0.0, 0.0),
    diagnostics_dir: str | Path | None = None,
) -> SafeDistanceFieldBundle:

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meshes = extract_exact_pipe_meshes(stage, collider_root_path, env_origin=env_origin)
    exact_mesh_digest = _mesh_digest(meshes)
    cache_id = make_cache_id(
        scene_path,
        {
            "algorithm_version": _FIELD_ALGORITHM_VERSION,
            "spec": spec.__dict__,
            "scene": scene_config,
        },
        exact_mesh_digest,
    )
    cache_npz = cache_dir / f"safe_distance_field_{cache_id}.npz"
    cache_json = cache_dir / f"safe_distance_field_{cache_id}.json"
    route_slot_by_id = {route_id: slot for slot, route_id in enumerate(spec.active_route_ids)}

    if cache_npz.exists() and cache_json.exists():
        with np.load(cache_npz, allow_pickle=False) as data:
            distances = data["distances_m"]
            reachable = data["reachable"]
            forbidden = data["forbidden"]
            clearance = data["clearance_m"]
            transition_masks = data["transition_masks"]
            interpolation_domain_bits = data["interpolation_domain_bits"]
        metadata = json.loads(cache_json.read_text(encoding="utf-8"))
        if metadata.get("cache_id") != cache_id:
            raise RuntimeError("distance-field cache metadata hash mismatch.")
        if metadata.get("exact_mesh_digest") != exact_mesh_digest:
            raise RuntimeError("exact distance fieldcache exact mesh hash does not match")
        if metadata.get("algorithm_version") != _FIELD_ALGORITHM_VERSION:
            raise RuntimeError("distance-field cache algorithm version mismatch.")
        if tuple(metadata.get("shape", ())) != spec.shape:
            raise RuntimeError("distance-field cache shape does not match the current workspace.")
        cached_route_ids = tuple(int(value) for value in metadata.get("active_route_ids", ()))
        if cached_route_ids != spec.active_route_ids:
            raise RuntimeError("distance-field cache active routes do not match the current configuration.")
        bundle = SafeDistanceFieldBundle(
            spec=spec,
            route_slot_by_id=route_slot_by_id,
            distances_m=distances,
            reachable=reachable,
            forbidden=forbidden,
            clearance_m=clearance,
            cache_id=cache_id,
            planned_lengths_m={},
            path_points_by_route={},
            transition_masks=transition_masks,
            interpolation_domain_bits=interpolation_domain_bits,
        )
    else:
        forbidden, clearance, obstacle_clearance = build_exact_mesh_forbidden_mask(meshes, spec)
        free_mask = ~forbidden
        transition_masks = build_exact_transition_masks(
            free_mask,
            obstacle_clearance,
            meshes,
            spec,
        )
        interpolation_domain_bits = build_interpolation_domain_bits(
            free_mask,
            transition_masks,
            obstacle_clearance_m=obstacle_clearance,
            meshes=meshes,
            spec=spec,
        )
        transition_masks = constrain_transitions_to_interpolation_domains(
            transition_masks,
            interpolation_domain_bits,
        )
        distances_list = []
        reachable_list = []
        for route_id in spec.active_route_ids:
            goal_seeds = _point_corner_seeds(
                goals_by_route[route_id],
                spec,
                free_mask,
                interpolation_domain_bits=interpolation_domain_bits,
                exact_meshes=meshes,
            )
            if not goal_seeds:
                raise RuntimeError(
                    f"route {route_id} exact goal has no safe free corner within 1.5 m: {goals_by_route[route_id]}"
                )
            raw_distance = dijkstra_distance_field(
                free_mask,
                goal_seeds[0][0],
                spec.resolution_m,
                initial_seeds=goal_seeds,
                transition_masks=transition_masks,
            )
            sanitized, reachable = _finite_sentinel(raw_distance, free_mask, spec)
            distances_list.append(sanitized)
            reachable_list.append(reachable)
        distances = np.stack(distances_list, axis=0)
        reachable = np.stack(reachable_list, axis=0)
        temporary_npz = cache_npz.with_suffix(".npz.tmp")
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(
                stream,
                distances_m=distances,
                reachable=reachable,
                forbidden=forbidden,
                clearance_m=clearance,
                transition_masks=transition_masks,
                interpolation_domain_bits=interpolation_domain_bits,
            )
        temporary_npz.replace(cache_npz)
        temporary_json = cache_json.with_suffix(".json.tmp")
        temporary_json.write_text(
            json.dumps(
                {
                    "cache_id": cache_id,
                    "algorithm_version": _FIELD_ALGORITHM_VERSION,
                    "exact_mesh_digest": exact_mesh_digest,
                    "shape": spec.shape,
                    "active_route_ids": spec.active_route_ids,
                    "spec": spec.__dict__,
                    "scene_config": scene_config,
                    "route_slot_by_id": route_slot_by_id,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_json.replace(cache_json)
        bundle = SafeDistanceFieldBundle(
            spec=spec,
            route_slot_by_id=route_slot_by_id,
            distances_m=distances,
            reachable=reachable,
            forbidden=forbidden,
            clearance_m=clearance,
            cache_id=cache_id,
            planned_lengths_m={},
            path_points_by_route={},
            transition_masks=transition_masks,
            interpolation_domain_bits=interpolation_domain_bits,
        )

    preflight_route_fields(
        bundle,
        starts_by_route,
        goals_by_route,
        diagnostics_dir=diagnostics_dir,
        exact_meshes=meshes,
    )
    return bundle


from typing import Any, Sequence


EXTENDED_SAFE_DISTANCE_ALGORITHM_VERSION = "base_double_domain_nearest_core_exact_mesh_r1"


_NEIGHBOR_OFFSETS: tuple[tuple[int, int, int, float], ...] = tuple(
    (dx, dy, dz, float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)
_CANONICAL_EDGE_OFFSETS: tuple[tuple[int, int, int, float], ...] = tuple(
    (dx, dy, dz, step)
    for dx, dy, dz, step in _NEIGHBOR_OFFSETS
    if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0)
)
_EDGE_SLOT_BY_OFFSET: dict[tuple[int, int, int], tuple[int, bool]] = {}
for _slot, (_dx, _dy, _dz, _) in enumerate(_CANONICAL_EDGE_OFFSETS):
    _EDGE_SLOT_BY_OFFSET[(_dx, _dy, _dz)] = (_slot, True)
    _EDGE_SLOT_BY_OFFSET[(-_dx, -_dy, -_dz)] = (_slot, False)
_TRANSITION_CODES = np.zeros(27, dtype=np.int8)
for _offset, (_slot, _forward) in _EDGE_SLOT_BY_OFFSET.items():
    _code_index = (_offset[0] + 1) * 9 + (_offset[1] + 1) * 3 + (_offset[2] + 1)
    _TRANSITION_CODES[_code_index] = (_slot + 1) if _forward else -(_slot + 1)


@dataclass(frozen=True)
class ExtendedSafeDistanceFieldSpec:

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    resolution_m: float = 0.5
    clearance_m: float = 0.75
    extension_clearance_m: float = 0.002
    active_route_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("base Dsafe must provide3D workspace lower/upper bounds")
        if any(float(lo) >= float(hi) for lo, hi in zip(self.lower, self.upper)):
            raise ValueError(f"base workspace invalid: lower={self.lower}, upper={self.upper}")
        if not np.isclose(float(self.resolution_m), 0.5, atol=1.0e-8):
            raise ValueError("resolution contract is fixed to 0.5 m.")
        if not np.isclose(float(self.clearance_m), 0.75, atol=1.0e-8):
            raise ValueError("core clearance contract is fixed to 0.75 m.")
        if float(self.extension_clearance_m) <= 0.0:
            raise ValueError("extended-domain exact edge certified clearance must be positive to prevent crossing obstacle interiors.")
        if float(self.extension_clearance_m) > float(self.clearance_m):
            raise ValueError("extended domainclearancemust not exceedcoreclearance")
        if tuple(self.active_route_ids) != (0, 1, 2, 3, 4, 5, 6):
            raise ValueError("base Dsafe supports exactly seven confirmed routes (0,1,2,3,4,5,6)")

    @property
    def shape(self) -> tuple[int, int, int]:
        spans = (np.asarray(self.upper, dtype=np.float64) - np.asarray(self.lower, dtype=np.float64)) / float(
            self.resolution_m
        )
        return tuple(int(np.ceil(value - 1.0e-8)) + 1 for value in spans)


@dataclass(frozen=True)
class _ExtensionGeometrySpec:

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    resolution_m: float
    clearance_m: float
    active_route_ids: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        spans = (np.asarray(self.upper, dtype=np.float64) - np.asarray(self.lower, dtype=np.float64)) / float(
            self.resolution_m
        )
        return tuple(int(np.ceil(value - 1.0e-8)) + 1 for value in spans)


@dataclass
class ExtendedSafeDistanceFieldBundle:

    spec: ExtendedSafeDistanceFieldSpec
    route_slot_by_id: dict[int, int]
    core_distances_m: np.ndarray
    core_reachable: np.ndarray
    extended_distances_m: np.ndarray
    extended_reachable: np.ndarray
    core_forbidden: np.ndarray
    clearance_m: np.ndarray
    core_transition_masks: np.ndarray
    core_domain_bits: np.ndarray
    extension_domain_bits: np.ndarray
    regression_distance_m: np.ndarray
    regression_source_flat: np.ndarray
    cache_id: str
    exact_mesh_digest: str
    planned_lengths_m: dict[int, float]
    path_points_by_route: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        route_shape = (len(self.route_slot_by_id),) + self.spec.shape
        for name, value in (
            ("core_distances_m", self.core_distances_m),
            ("core_reachable", self.core_reachable),
            ("extended_distances_m", self.extended_distances_m),
            ("extended_reachable", self.extended_reachable),
        ):
            if tuple(value.shape) != route_shape:
                raise ValueError(f"{name} shape mismatch: expected={route_shape}, actual={value.shape}")
        for name, value in (
            ("core_forbidden", self.core_forbidden),
            ("clearance_m", self.clearance_m),
            ("core_domain_bits", self.core_domain_bits),
            ("extension_domain_bits", self.extension_domain_bits),
            ("regression_distance_m", self.regression_distance_m),
            ("regression_source_flat", self.regression_source_flat),
        ):
            if tuple(value.shape) != self.spec.shape:
                raise ValueError(f"{name} does not match the grid shape.")
        expected_edges = (len(_CANONICAL_EDGE_OFFSETS),) + self.spec.shape
        if tuple(self.core_transition_masks.shape) != expected_edges:
            raise ValueError("core_transition_masks shape mismatch.")
        if not np.isfinite(self.core_distances_m).all() or not np.isfinite(self.extended_distances_m).all():
            raise ValueError("base distance fieldcachecontains NaN/Inf")

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        lower = np.asarray(self.spec.lower, dtype=np.float64)
        return np.rint((points - lower) / float(self.spec.resolution_m)).astype(np.int64)

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.float64)
        return np.asarray(self.spec.lower, dtype=np.float64) + indices * float(self.spec.resolution_m)

    def to_torch(self, device: Any) -> "TorchExtendedSafeDistanceField":

        import torch

        route_slots = np.full(max(self.route_slot_by_id) + 1, -1, dtype=np.int64)
        for route_id, slot in self.route_slot_by_id.items():
            route_slots[int(route_id)] = int(slot)
        return TorchExtendedSafeDistanceField(
            core_distances_m=torch.as_tensor(self.core_distances_m, dtype=torch.float32, device=device),
            core_reachable=torch.as_tensor(self.core_reachable, dtype=torch.bool, device=device),
            extended_distances_m=torch.as_tensor(self.extended_distances_m, dtype=torch.float32, device=device),
            extended_reachable=torch.as_tensor(self.extended_reachable, dtype=torch.bool, device=device),
            core_domain_bits=torch.as_tensor(self.core_domain_bits, dtype=torch.uint8, device=device),
            extension_domain_bits=torch.as_tensor(self.extension_domain_bits, dtype=torch.uint8, device=device),
            lower=torch.tensor(self.spec.lower, dtype=torch.float32, device=device),
            resolution_m=float(self.spec.resolution_m),
            route_slot_by_id=torch.as_tensor(route_slots, dtype=torch.long, device=device),
            cache_id=self.cache_id,
        )


@dataclass
class TorchExtendedSafeDistanceField:

    core_distances_m: Any
    core_reachable: Any
    extended_distances_m: Any
    extended_reachable: Any
    core_domain_bits: Any
    extension_domain_bits: Any
    lower: Any
    resolution_m: float
    route_slot_by_id: Any
    cache_id: str

    def _lookup_field(self, positions, slots, distances, reachable, domain_bits):

        import torch

        shape = distances.shape[1:]
        shape_tensor = torch.tensor(shape, dtype=torch.long, device=positions.device)
        coordinate = (positions - self.lower.unsqueeze(0)) / float(self.resolution_m)
        inside = torch.all(
            torch.logical_and(coordinate >= 0.0, coordinate <= (shape_tensor.to(torch.float32) - 1.0)), dim=1
        )
        coordinate = torch.minimum(coordinate.clamp_min(0.0), shape_tensor.to(torch.float32) - 1.0)
        index0 = torch.floor(coordinate).to(torch.long)
        index1 = torch.minimum(index0 + 1, shape_tensor - 1)
        weight = coordinate - index0.to(coordinate.dtype)

        values = []
        valid_corners = []
        weights = []
        for dx, dy, dz in (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ):
            ix = index1[:, 0] if dx else index0[:, 0]
            iy = index1[:, 1] if dy else index0[:, 1]
            iz = index1[:, 2] if dz else index0[:, 2]
            values.append(distances[slots, ix, iy, iz])
            valid_corners.append(reachable[slots, ix, iy, iz])
            wx = weight[:, 0] if dx else 1.0 - weight[:, 0]
            wy = weight[:, 1] if dy else 1.0 - weight[:, 1]
            wz = weight[:, 2] if dz else 1.0 - weight[:, 2]
            weights.append(wx * wy * wz)
        values_t = torch.stack(values, dim=1)
        valid_t = torch.stack(valid_corners, dim=1)
        weights_t = torch.stack(weights, dim=1)
        required_corner = weights_t > 1.0e-7
        corners_valid = torch.logical_or(~required_corner, valid_t).all(dim=1)
        value = torch.sum(values_t * weights_t, dim=1)

        active_axes = weight > 1.0e-7
        active_code = (
            active_axes[:, 0].to(torch.long)
            + 2 * active_axes[:, 1].to(torch.long)
            + 4 * active_axes[:, 2].to(torch.long)
        )
        bits = domain_bits[index0[:, 0], index0[:, 1], index0[:, 2]].to(torch.long)
        required_bit = torch.bitwise_left_shift(torch.ones_like(active_code), active_code)
        domain_valid = torch.bitwise_and(bits, required_bit) != 0
        valid = inside & corners_valid & domain_valid
        return torch.where(valid, value, torch.zeros_like(value)), valid

    def lookup(self, positions_local_m, route_ids):

        import torch

        if positions_local_m.ndim != 2 or positions_local_m.shape[1] != 3:
            raise ValueError("base Dsafe query positionsmust be [N,3]")
        route_ids = route_ids.to(torch.long)
        route_in_range = (route_ids >= 0) & (route_ids < self.route_slot_by_id.numel())
        safe_ids = route_ids.clamp(0, self.route_slot_by_id.numel() - 1)
        slots = self.route_slot_by_id[safe_ids]
        route_valid = route_in_range & (slots >= 0) & (slots < self.core_distances_m.shape[0])
        safe_slots = slots.clamp(0, self.core_distances_m.shape[0] - 1)

        core_value, core_valid = self._lookup_field(
            positions_local_m, safe_slots, self.core_distances_m, self.core_reachable, self.core_domain_bits
        )
        ext_value, ext_valid = self._lookup_field(
            positions_local_m,
            safe_slots,
            self.extended_distances_m,
            self.extended_reachable,
            self.extension_domain_bits,
        )
        valid = route_valid & (core_valid | ext_valid)
        selected = torch.where(core_valid, core_value, ext_value)
        return torch.where(valid, selected, torch.zeros_like(selected)), valid


def _transition_allowed(
    masks: np.ndarray | None,
    current: tuple[int, int, int],
    neighbor: tuple[int, int, int],
    offset: tuple[int, int, int],
) -> bool:
    if masks is None:
        for x in (current[0], neighbor[0]) if offset[0] else (current[0],):
            for y in (current[1], neighbor[1]) if offset[1] else (current[1],):
                for z in (current[2], neighbor[2]) if offset[2] else (current[2],):
                    yield_index = (x, y, z)
                    del yield_index
        return True
    slot, forward = _EDGE_SLOT_BY_OFFSET[offset]
    source = current if forward else neighbor
    return bool(masks[(slot,) + source])


def _multi_source_regression_python(
    free_mask: np.ndarray,
    seed_nodes_flat: np.ndarray,
    resolution_m: float,
    transition_masks: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:

    shape = free_mask.shape
    distance = np.full(shape, np.inf, dtype=np.float32)
    source = np.full(shape, -1, dtype=np.int32)
    heap: list[tuple[float, int, int]] = []
    for seed_flat in np.asarray(seed_nodes_flat, dtype=np.int64):
        index = tuple(int(v) for v in np.unravel_index(int(seed_flat), shape))
        if not free_mask[index]:
            continue
        distance[index] = 0.0
        source[index] = int(seed_flat)
        heapq.heappush(heap, (0.0, int(seed_flat), int(seed_flat)))
    while heap:
        current_distance, _tie_source, current_flat = heapq.heappop(heap)
        current = tuple(int(v) for v in np.unravel_index(current_flat, shape))
        if current_distance > float(distance[current]) + 1.0e-7:
            continue
        current_source = int(source[current])
        for dx, dy, dz, step in _NEIGHBOR_OFFSETS:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if any(value < 0 or value >= shape[axis] for axis, value in enumerate(neighbor)):
                continue
            if not free_mask[neighbor]:
                continue
            if transition_masks is not None:
                if not _transition_allowed(transition_masks, current, neighbor, (dx, dy, dz)):
                    continue
            else:
                support_clear = True
                for x in (current[0], neighbor[0]) if dx else (current[0],):
                    for y in (current[1], neighbor[1]) if dy else (current[1],):
                        for z in (current[2], neighbor[2]) if dz else (current[2],):
                            if not free_mask[x, y, z]:
                                support_clear = False
                                break
                        if not support_clear:
                            break
                    if not support_clear:
                        break
                if not support_clear:
                    continue
            candidate = current_distance + float(resolution_m) * step
            old = float(distance[neighbor])
            old_source = int(source[neighbor])
            if candidate > old + 1.0e-7:
                continue
            if abs(candidate - old) <= 1.0e-7 and old_source >= 0 and current_source >= old_source:
                continue
            distance[neighbor] = candidate
            source[neighbor] = current_source
            neighbor_flat = int(np.ravel_multi_index(neighbor, shape))
            heapq.heappush(heap, (candidate, current_source, neighbor_flat))
    return distance, source


try:
    import numba
except ImportError:
    numba = None


if numba is not None:

    @numba.njit(cache=False)
    def _multi_source_regression_numba(
        free_flat,
        nx,
        ny,
        nz,
        seed_nodes,
        resolution_m,
        transition_masks_flat,
        transition_codes,
        use_transition_masks,
    ):

        node_count = free_flat.size
        distances = np.full(node_count, np.inf, dtype=np.float32)
        sources = np.full(node_count, -1, dtype=np.int32)
        positions = np.full(node_count, -1, dtype=np.int32)
        heap_nodes = np.empty(node_count, dtype=np.int32)
        heap_values = np.empty(node_count, dtype=np.float32)
        heap_size = 0
        for seed_i in range(seed_nodes.size):
            seed = seed_nodes[seed_i]
            if not free_flat[seed] or distances[seed] == 0.0:
                continue
            distances[seed] = 0.0
            sources[seed] = seed
            positions[seed] = heap_size
            heap_nodes[heap_size] = seed
            heap_values[heap_size] = 0.0
            heap_size += 1

        xy_stride = ny * nz
        sqrt2 = np.sqrt(2.0)
        sqrt3 = np.sqrt(3.0)
        while heap_size > 0:
            current_node = heap_nodes[0]
            current_value = heap_values[0]
            current_source = sources[current_node]
            positions[current_node] = -1
            heap_size -= 1
            if heap_size > 0:
                moved_node = heap_nodes[heap_size]
                heap_nodes[0] = moved_node
                heap_values[0] = heap_values[heap_size]
                positions[moved_node] = 0
                parent = 0
                while True:
                    left = parent * 2 + 1
                    if left >= heap_size:
                        break
                    right = left + 1
                    child = left
                    if right < heap_size and heap_values[right] < heap_values[left]:
                        child = right
                    if heap_values[parent] <= heap_values[child]:
                        break
                    parent_node = heap_nodes[parent]
                    child_node = heap_nodes[child]
                    heap_nodes[parent], heap_nodes[child] = child_node, parent_node
                    heap_values[parent], heap_values[child] = heap_values[child], heap_values[parent]
                    positions[parent_node], positions[child_node] = child, parent
                    parent = child

            x = current_node // xy_stride
            remainder = current_node - x * xy_stride
            y = remainder // nz
            z = remainder - y * nz
            for dx in (-1, 0, 1):
                xx = x + dx
                if xx < 0 or xx >= nx:
                    continue
                for dy in (-1, 0, 1):
                    yy = y + dy
                    if yy < 0 or yy >= ny:
                        continue
                    for dz in (-1, 0, 1):
                        zz = z + dz
                        if zz < 0 or zz >= nz or (dx == 0 and dy == 0 and dz == 0):
                            continue
                        neighbor = xx * xy_stride + yy * nz + zz
                        if not free_flat[neighbor]:
                            continue
                        if use_transition_masks:
                            code_index = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
                            transition_code = transition_codes[code_index]
                            if transition_code > 0:
                                edge_slot = transition_code - 1
                                edge_source = current_node
                            else:
                                edge_slot = -transition_code - 1
                                edge_source = neighbor
                            if not transition_masks_flat[edge_slot, edge_source]:
                                continue
                        else:
                            support_clear = True
                            x_count = 2 if dx != 0 else 1
                            y_count = 2 if dy != 0 else 1
                            z_count = 2 if dz != 0 else 1
                            for xi in range(x_count):
                                cx = x + (dx if xi == 1 else 0)
                                for yi in range(y_count):
                                    cy = y + (dy if yi == 1 else 0)
                                    for zi in range(z_count):
                                        cz = z + (dz if zi == 1 else 0)
                                        if not free_flat[cx * xy_stride + cy * nz + cz]:
                                            support_clear = False
                                            break
                                    if not support_clear:
                                        break
                                if not support_clear:
                                    break
                            if not support_clear:
                                continue
                        squared = dx * dx + dy * dy + dz * dz
                        step = 1.0 if squared == 1 else (sqrt2 if squared == 2 else sqrt3)
                        candidate = current_value + resolution_m * step
                        old = distances[neighbor]
                        if candidate > old + 1.0e-6:
                            continue
                        if np.abs(candidate - old) <= 1.0e-6 and sources[neighbor] >= 0 and current_source >= sources[neighbor]:
                            continue
                        distances[neighbor] = candidate
                        sources[neighbor] = current_source
                        heap_position = positions[neighbor]
                        if heap_position < 0:
                            heap_position = heap_size
                            heap_size += 1
                            heap_nodes[heap_position] = neighbor
                            positions[neighbor] = heap_position
                        heap_values[heap_position] = candidate
                        while heap_position > 0:
                            parent = (heap_position - 1) // 2
                            if heap_values[parent] <= heap_values[heap_position]:
                                break
                            parent_node = heap_nodes[parent]
                            child_node = heap_nodes[heap_position]
                            heap_nodes[parent], heap_nodes[heap_position] = child_node, parent_node
                            heap_values[parent], heap_values[heap_position] = heap_values[heap_position], heap_values[parent]
                            positions[parent_node], positions[child_node] = heap_position, parent
                            heap_position = parent
        return distances, sources


def multi_source_nearest_core_regression(
    propagation_mask: np.ndarray,
    seed_nodes_flat: np.ndarray,
    resolution_m: float,
    transition_masks: np.ndarray | None = None,
    *,
    max_python_nodes: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:

    propagation_mask = np.asarray(propagation_mask, dtype=np.bool_)
    if propagation_mask.ndim != 3:
        raise ValueError("propagation_mask must be a 3D bool array.")
    seed_nodes_flat = np.asarray(seed_nodes_flat, dtype=np.int32).reshape(-1)
    if seed_nodes_flat.size == 0:
        raise ValueError("nearest-core regression requires at least one boundary seed.")
    if transition_masks is not None:
        transition_masks = np.asarray(transition_masks, dtype=np.bool_)
        expected = (len(_CANONICAL_EDGE_OFFSETS),) + propagation_mask.shape
        if tuple(transition_masks.shape) != expected:
            raise ValueError(f"transition_masks shape mismatch: expected={expected}, actual={transition_masks.shape}")

    if numba is not None:
        masks_flat = (
            np.ascontiguousarray(transition_masks.reshape(len(_CANONICAL_EDGE_OFFSETS), -1))
            if transition_masks is not None
            else np.zeros((1, 1), dtype=np.bool_)
        )
        distances, sources = _multi_source_regression_numba(
            np.ascontiguousarray(propagation_mask.reshape(-1)),
            int(propagation_mask.shape[0]),
            int(propagation_mask.shape[1]),
            int(propagation_mask.shape[2]),
            seed_nodes_flat,
            float(resolution_m),
            masks_flat,
            _TRANSITION_CODES,
            transition_masks is not None,
        )
        return distances.reshape(propagation_mask.shape), sources.reshape(propagation_mask.shape)
    if propagation_mask.size > int(max_python_nodes):
        raise RuntimeError("building the large-scale regression field requires numba; refusing to fall back to an hours-long Python loop.")
    return _multi_source_regression_python(
        propagation_mask, seed_nodes_flat, float(resolution_m), transition_masks
    )


def _core_boundary_mask(core_free: np.ndarray, extension_free: np.ndarray) -> np.ndarray:

    from scipy import ndimage

    outer = extension_free & ~core_free
    return core_free & ndimage.binary_dilation(outer, structure=np.ones((3, 3, 3), dtype=bool))


def _finite_sentinel_for_extended(
    values: np.ndarray, reachable: np.ndarray, spec: ExtendedSafeDistanceFieldSpec
) -> np.ndarray:

    span = np.asarray(spec.upper, dtype=np.float64) - np.asarray(spec.lower, dtype=np.float64)
    sentinel = float(np.linalg.norm(span) * 4.0 + 1.0)
    return np.where(reachable, values, sentinel).astype(np.float32)


def _build_extended_route_fields(
    core_distances: np.ndarray,
    core_reachable: np.ndarray,
    core_free: np.ndarray,
    extension_free: np.ndarray,
    regression_distance: np.ndarray,
    regression_source: np.ndarray,
    spec: ExtendedSafeDistanceFieldSpec,
) -> tuple[np.ndarray, np.ndarray]:

    flat_source = regression_source.reshape(-1)
    source_valid = flat_source >= 0
    safe_source = np.maximum(flat_source, 0)
    regression_flat = regression_distance.reshape(-1)
    route_values: list[np.ndarray] = []
    route_valid: list[np.ndarray] = []
    for slot in range(core_distances.shape[0]):
        core_flat = core_distances[slot].reshape(-1)
        core_reachable_flat = core_reachable[slot].reshape(-1)
        valid = source_valid & extension_free.reshape(-1) & core_reachable_flat[safe_source]
        value = core_flat[safe_source] + regression_flat
        value[core_free.reshape(-1)] = core_distances[slot].reshape(-1)[core_free.reshape(-1)]
        valid[core_free.reshape(-1)] = core_reachable[slot].reshape(-1)[core_free.reshape(-1)]
        value = _finite_sentinel_for_extended(value.reshape(spec.shape), valid.reshape(spec.shape), spec)
        route_values.append(value)
        route_valid.append(valid.reshape(spec.shape))
    return np.stack(route_values, axis=0), np.stack(route_valid, axis=0)


def _audit_double_domain(bundle: ExtendedSafeDistanceFieldBundle) -> dict[str, object]:

    core_free = ~bundle.core_forbidden
    extension_nodes = bundle.regression_source_flat >= 0
    source = bundle.regression_source_flat[extension_nodes]
    source_in_range = (source >= 0) & (source < int(np.prod(bundle.spec.shape)))
    source_core = np.zeros(source.shape, dtype=bool)
    if source.size:
        source_core[source_in_range] = core_free.reshape(-1)[source[source_in_range]]
    if not source_in_range.all() or not source_core.all():
        raise RuntimeError("double-domain regression source contains out-of-bounds or non-core nodes.")
    for slot in range(bundle.core_distances_m.shape[0]):
        mask = core_free & bundle.core_reachable[slot]
        if mask.any() and not np.allclose(
            bundle.extended_distances_m[slot][mask], bundle.core_distances_m[slot][mask], atol=1.0e-5
        ):
            raise RuntimeError(f"route slot {slot} extended field changed Dcore inside the core region.")
    if not np.isfinite(bundle.regression_distance_m[extension_nodes]).all():
        raise RuntimeError("regression distances contain NaN/Inf.")
    return {
        "algorithm_version": EXTENDED_SAFE_DISTANCE_ALGORITHM_VERSION,
        "core_node_count": int(core_free.sum()),
        "extended_source_node_count": int(extension_nodes.sum()),
        "max_regression_distance_m": float(np.max(bundle.regression_distance_m[extension_nodes]))
        if extension_nodes.any()
        else 0.0,
        "finite": True,
        "core_identity": True,
    }


def build_extended_safe_distance_field(
    *,
    stage,
    collider_root_path: str,
    scene_path: str | Path,
    spec: ExtendedSafeDistanceFieldSpec,
    scene_config: dict[str, Any],
    cache_dir: str | Path,
    starts_by_route: dict[int, Sequence[float]],
    goals_by_route: dict[int, Sequence[float]],
    env_origin: Sequence[float] = (0.0, 0.0, 0.0),
    diagnostics_dir: str | Path | None = None,
) -> ExtendedSafeDistanceFieldBundle:

    from . import safe_distance_field as exact

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_dir is not None:
        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    meshes = extract_exact_pipe_meshes(stage, collider_root_path, env_origin=env_origin)
    exact_mesh_digest = _mesh_digest(meshes)
    cache_id = make_cache_id(
        scene_path,
        {
            "algorithm_version": EXTENDED_SAFE_DISTANCE_ALGORITHM_VERSION,
            "spec": spec.__dict__,
            "scene": scene_config,
        },
        exact_mesh_digest,
    )
    cache_npz = cache_dir / f"base_safe_distance_field_{cache_id}.npz"
    cache_json = cache_dir / f"base_safe_distance_field_{cache_id}.json"
    route_slot_by_id = {route_id: slot for slot, route_id in enumerate(spec.active_route_ids)}

    if cache_npz.exists() and cache_json.exists():
        metadata = json.loads(cache_json.read_text(encoding="utf-8"))
        if metadata.get("cache_id") != cache_id:
            raise RuntimeError("Dsafe cache id does not match the current exact geometry.")
        if metadata.get("algorithm_version") != EXTENDED_SAFE_DISTANCE_ALGORITHM_VERSION:
            raise RuntimeError("Dsafe algorithm version mismatch; refusing to read a stale cache.")
        if metadata.get("exact_mesh_digest") != exact_mesh_digest:
            raise RuntimeError("base Dsafe exact mesh digest does not match")
        if tuple(metadata.get("shape", ())) != spec.shape:
            raise RuntimeError("base Dsafe cache shape does not match")
        with np.load(cache_npz, allow_pickle=False) as data:
            bundle = ExtendedSafeDistanceFieldBundle(
                spec=spec,
                route_slot_by_id=route_slot_by_id,
                core_distances_m=data["core_distances_m"],
                core_reachable=data["core_reachable"],
                extended_distances_m=data["extended_distances_m"],
                extended_reachable=data["extended_reachable"],
                core_forbidden=data["core_forbidden"],
                clearance_m=data["clearance_m"],
                core_transition_masks=data["core_transition_masks"],
                core_domain_bits=data["core_domain_bits"],
                extension_domain_bits=data["extension_domain_bits"],
                regression_distance_m=data["regression_distance_m"],
                regression_source_flat=data["regression_source_flat"],
                cache_id=cache_id,
                exact_mesh_digest=exact_mesh_digest,
                planned_lengths_m={},
                path_points_by_route={},
            )
    else:
        core_forbidden, clearance, obstacle_clearance = build_exact_mesh_forbidden_mask(meshes, spec)
        core_free = ~core_forbidden
        core_transition_masks = build_exact_transition_masks(
            core_free, obstacle_clearance, meshes, spec
        )
        core_domain_bits = build_interpolation_domain_bits(
            core_free,
            core_transition_masks,
            obstacle_clearance_m=obstacle_clearance,
            meshes=meshes,
            spec=spec,
        )
        core_transition_masks = constrain_transitions_to_interpolation_domains(
            core_transition_masks, core_domain_bits
        )

        core_distances_list: list[np.ndarray] = []
        core_reachable_list: list[np.ndarray] = []
        for route_id in spec.active_route_ids:
            goal_seeds = _point_corner_seeds(
                goals_by_route[route_id],
                spec,
                core_free,
                interpolation_domain_bits=core_domain_bits,
                exact_meshes=meshes,
            )
            if not goal_seeds:
                raise RuntimeError(
                    f"route {route_id} goal has no core corner with 0.75 m exact clearance."
                )
            raw = dijkstra_distance_field(
                core_free,
                goal_seeds[0][0],
                spec.resolution_m,
                initial_seeds=goal_seeds,
                transition_masks=core_transition_masks,
            )
            sanitized, reachable = _finite_sentinel(raw, core_free, spec)
            core_distances_list.append(sanitized)
            core_reachable_list.append(reachable)
        core_distances = np.stack(core_distances_list, axis=0)
        core_reachable = np.stack(core_reachable_list, axis=0)

        extension_spec = _ExtensionGeometrySpec(
            lower=spec.lower,
            upper=spec.upper,
            resolution_m=spec.resolution_m,
            clearance_m=spec.extension_clearance_m,
            active_route_ids=spec.active_route_ids,
        )
        extension_free = obstacle_clearance >= float(spec.extension_clearance_m) - 1.0e-6
        extension_transition_masks = build_exact_transition_masks(
            extension_free, obstacle_clearance, meshes, extension_spec
        )
        extension_domain_bits = build_interpolation_domain_bits(
            extension_free,
            extension_transition_masks,
            obstacle_clearance_m=obstacle_clearance,
            meshes=meshes,
            spec=extension_spec,
        )
        extension_transition_masks = constrain_transitions_to_interpolation_domains(
            extension_transition_masks, extension_domain_bits
        )
        boundary_seeds = _core_boundary_mask(core_free, extension_free)
        propagation_mask = extension_free & (~core_free | boundary_seeds)
        seed_nodes = np.flatnonzero(boundary_seeds.reshape(-1)).astype(np.int32)
        regression_distance, regression_source = multi_source_nearest_core_regression(
            propagation_mask,
            seed_nodes,
            spec.resolution_m,
            transition_masks=extension_transition_masks,
        )
        flat_indices = np.arange(int(np.prod(spec.shape)), dtype=np.int32).reshape(spec.shape)
        regression_distance[core_free] = 0.0
        regression_source[core_free] = flat_indices[core_free]
        extended_distances, extended_reachable = _build_extended_route_fields(
            core_distances,
            core_reachable,
            core_free,
            extension_free,
            regression_distance,
            regression_source,
            spec,
        )
        finite_regression = np.where(np.isfinite(regression_distance), regression_distance, 0.0).astype(np.float32)

        bundle = ExtendedSafeDistanceFieldBundle(
            spec=spec,
            route_slot_by_id=route_slot_by_id,
            core_distances_m=core_distances,
            core_reachable=core_reachable,
            extended_distances_m=extended_distances,
            extended_reachable=extended_reachable,
            core_forbidden=core_forbidden,
            clearance_m=clearance,
            core_transition_masks=core_transition_masks,
            core_domain_bits=core_domain_bits,
            extension_domain_bits=extension_domain_bits,
            regression_distance_m=finite_regression,
            regression_source_flat=regression_source,
            cache_id=cache_id,
            exact_mesh_digest=exact_mesh_digest,
            planned_lengths_m={},
            path_points_by_route={},
        )
        audit = _audit_double_domain(bundle)
        temporary_npz = cache_npz.with_suffix(".npz.tmp")
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(
                stream,
                core_distances_m=bundle.core_distances_m,
                core_reachable=bundle.core_reachable,
                extended_distances_m=bundle.extended_distances_m,
                extended_reachable=bundle.extended_reachable,
                core_forbidden=bundle.core_forbidden,
                clearance_m=bundle.clearance_m,
                core_transition_masks=bundle.core_transition_masks,
                core_domain_bits=bundle.core_domain_bits,
                extension_domain_bits=bundle.extension_domain_bits,
                regression_distance_m=bundle.regression_distance_m,
                regression_source_flat=bundle.regression_source_flat,
            )
        temporary_npz.replace(cache_npz)
        temporary_json = cache_json.with_suffix(".json.tmp")
        temporary_json.write_text(
            json.dumps(
                {
                    "cache_id": cache_id,
                    "algorithm_version": EXTENDED_SAFE_DISTANCE_ALGORITHM_VERSION,
                    "exact_mesh_digest": exact_mesh_digest,
                    "shape": spec.shape,
                    "spec": spec.__dict__,
                    "scene_config": scene_config,
                    "route_slot_by_id": route_slot_by_id,
                    "double_domain_audit": audit,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_json.replace(cache_json)

    _audit_double_domain(bundle)
    preflight_bundle = SafeDistanceFieldBundle(
        spec=spec,
        route_slot_by_id=route_slot_by_id,
        distances_m=bundle.core_distances_m,
        reachable=bundle.core_reachable,
        forbidden=bundle.core_forbidden,
        clearance_m=bundle.clearance_m,
        cache_id=bundle.cache_id,
        planned_lengths_m={},
        path_points_by_route={},
        transition_masks=bundle.core_transition_masks,
        interpolation_domain_bits=bundle.core_domain_bits,
    )
    preflight_route_fields(
        preflight_bundle,
        starts_by_route,
        goals_by_route,
        diagnostics_dir=diagnostics_dir,
        exact_meshes=meshes,
    )
    bundle.planned_lengths_m = dict(preflight_bundle.planned_lengths_m)
    bundle.path_points_by_route = dict(preflight_bundle.path_points_by_route)
    return bundle
