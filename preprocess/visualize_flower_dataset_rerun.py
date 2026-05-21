"""
Visualize a FLOWER / CALVIN-style disk dataset with Rerun.

This script is intended for converted datasets that contain:
  - training/ or validation/episode_XXXXXXX.npz
  - ep_start_end_ids.npy
  - <lang_folder>/auto_lang_ann.npy

Each timestep file is expected to include some or all of:
  - rgb_static
  - rgb_gripper
  - robot_obs
  - rel_actions
  - scene_obs

Example:
    python preprocess/visualize_flower_dataset_rerun.py \
        --dataset_root D:\\your_dataset \
        --split training \
        --episode_index 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("rerun"):
        raise ModuleNotFoundError(
            "Failed to import the Rerun SDK. "
            "You likely installed the unrelated 'rerun' package instead of the robotics SDK. "
            "For Python 3.9, install 'rerun-sdk==0.26.1'. "
            "For Python 3.10+, install the latest 'rerun-sdk'."
        ) from exc
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Converted FLOWER dataset root containing training/ and validation/.",
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="training",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="If set, visualize only one episode using ep_start_end_ids.npy.",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Global start step within the split when --episode_index is not used.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps to visualize when --episode_index is not used.",
    )
    parser.add_argument(
        "--step_stride",
        type=int,
        default=1,
        help="Load every Nth step to reduce viewer load.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Synthetic seconds timeline used alongside the integer step timeline.",
    )
    parser.add_argument(
        "--lang_folder",
        default="lang_clip_resnet50",
        help="Language annotation folder name inside the split directory.",
    )
    parser.add_argument(
        "--headless_output",
        type=Path,
        default=None,
        help=(
            "Optional .rrd output path for headless environments. "
            "If omitted and no display server is available, the script auto-saves to a default .rrd file."
        ),
    )
    rr.script_add_args(parser)
    return parser.parse_args()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_pickle_compatible_npy(path: Path) -> Dict[str, Any]:
    try:
        return np.load(path, allow_pickle=True).item()
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise
        sys.modules.setdefault("numpy._core", np.core)
        return np.load(path, allow_pickle=True).item()


def load_language_data(path: Path) -> Dict[str, Any]:
    json_path = path.with_suffix(".json")
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return load_pickle_compatible_npy(path)


def load_boundaries(split_dir: Path) -> np.ndarray:
    boundary_path = split_dir / "ep_start_end_ids.npy"
    if not boundary_path.is_file():
        raise FileNotFoundError(f"Missing episode boundary file: {boundary_path}")
    boundaries = np.load(boundary_path)
    boundaries = np.asarray(boundaries, dtype=np.int64).reshape(-1, 2)
    return boundaries


def load_step_file(step_path: Path) -> Dict[str, np.ndarray]:
    with np.load(step_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def step_file(split_dir: Path, step_idx: int) -> Path:
    return split_dir / f"episode_{step_idx:07d}.npz"


def resolve_step_indices(
    split_dir: Path,
    boundaries: np.ndarray,
    *,
    episode_index: Optional[int],
    start_step: int,
    max_steps: Optional[int],
    step_stride: int,
) -> List[int]:
    if step_stride <= 0:
        raise ValueError(f"step_stride must be >= 1, got {step_stride}")

    if episode_index is not None:
        if episode_index < 0 or episode_index >= len(boundaries):
            raise IndexError(f"episode_index {episode_index} is out of range for {len(boundaries)} episodes")
        episode_start, episode_end = boundaries[episode_index]
        return list(range(int(episode_start), int(episode_end), step_stride))

    total_steps = len(list(split_dir.glob("episode_*.npz")))
    if start_step < 0 or start_step >= total_steps:
        raise IndexError(f"start_step {start_step} is out of range for {total_steps} steps")
    end_step = total_steps if max_steps is None else min(total_steps, start_step + max_steps)
    return list(range(start_step, end_step, step_stride))


def build_episode_annotation_lookup(split_dir: Path, lang_folder: str) -> Dict[int, Dict[str, Any]]:
    lang_path = split_dir / lang_folder / "auto_lang_ann.npy"
    if not lang_path.exists() and not lang_path.with_suffix(".json").exists():
        return {}

    raw = load_language_data(lang_path)
    language = raw.get("language", {})
    info = raw.get("info", {})

    annotations = language.get("ann", [])
    tasks = language.get("task", [])
    ranges = info.get("indx", [])

    lookup: Dict[int, Dict[str, Any]] = {}
    for ep_idx, episode_range in enumerate(ranges):
        if len(episode_range) != 2:
            continue
        start, end = int(episode_range[0]), int(episode_range[1])
        text = as_text(annotations[ep_idx]) if ep_idx < len(annotations) else ""
        task = as_text(tasks[ep_idx]) if ep_idx < len(tasks) else ""
        lookup[start] = {
            "episode_index": ep_idx,
            "start": start,
            "end": end,
            "task": task,
            "text": text,
        }
    return lookup


def build_series_views(prefix: str, label: str, dim: int) -> List[rrb.TimeSeriesView]:
    views = [rrb.TimeSeriesView(name=f"{label}[{idx}]", origin=f"/{prefix}/d{idx}") for idx in range(dim)]
    views.append(rrb.TimeSeriesView(name=f"{label} Norm", origin=f"/{prefix}_norm"))
    return views


def setup_blueprint(example_step: Dict[str, np.ndarray]) -> rrb.Blueprint:
    sections: List[Any] = []

    camera_views: List[Any] = []
    if "rgb_static" in example_step:
        camera_views.append(rrb.Spatial2DView(name="Static Cam", origin="/rgb_static"))
    if "rgb_gripper" in example_step:
        camera_views.append(rrb.Spatial2DView(name="Gripper Cam", origin="/rgb_gripper"))
    if camera_views:
        sections.append(rrb.Horizontal(*camera_views, name="Cameras"))

    sections.append(
        rrb.Horizontal(
            rrb.TextLogView(origin="/language", name="Language"),
            rrb.TextLogView(origin="/episode", name="Episodes"),
            name="Annotations",
        )
    )

    if "robot_obs" in example_step:
        robot_dim = int(np.asarray(example_step["robot_obs"]).reshape(-1).shape[0])
        sections.append(
            rrb.Grid(
                *build_series_views("robot_obs", "Robot", robot_dim),
                grid_columns=min(4, robot_dim + 1),
                name="Robot Obs",
            )
        )

    if "rel_actions" in example_step:
        action_dim = int(np.asarray(example_step["rel_actions"]).reshape(-1).shape[0])
        sections.append(
            rrb.Grid(
                *build_series_views("rel_actions", "Action", action_dim),
                grid_columns=min(4, action_dim + 1),
                name="Actions",
            )
        )

    if "scene_obs" in example_step:
        scene_dim = int(np.asarray(example_step["scene_obs"]).reshape(-1).shape[0])
        if scene_dim > 0:
            sections.append(
                rrb.Grid(
                    *build_series_views("scene_obs", "Scene", scene_dim),
                    grid_columns=min(4, scene_dim + 1),
                    name="Scene Obs",
                )
            )

    if not sections:
        sections.append(rrb.TextLogView(origin="/episode", name="Episodes"))

    return rrb.Blueprint(rrb.Vertical(*sections))


def is_headless_environment() -> bool:
    return not any(
        os.environ.get(name)
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET")
    )


def has_explicit_rerun_sink_args(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, field, None)
        for field in ("connect", "save", "serve", "stdout")
    )


def default_headless_output_path(
    dataset_root: Path,
    split: str,
    *,
    episode_index: Optional[int],
    start_step: int,
    max_steps: Optional[int],
) -> Path:
    if episode_index is not None:
        suffix = f"{split}_episode_{episode_index:04d}"
    else:
        end_step_desc = "end" if max_steps is None else str(start_step + max_steps)
        suffix = f"{split}_steps_{start_step}_{end_step_desc}"
    return dataset_root / f"flower_rerun_{suffix}.rrd"


def configure_rerun(
    args: argparse.Namespace,
    *,
    blueprint: rrb.Blueprint,
) -> bool:
    app_id = "flower_dataset_visualization"
    headless = is_headless_environment()

    if headless and not has_explicit_rerun_sink_args(args):
        output_path = args.headless_output or default_headless_output_path(
            args.dataset_root,
            args.split,
            episode_index=args.episode_index,
            start_step=args.start_step,
            max_steps=args.max_steps,
        )
        rr.init(app_id, spawn=False)
        rr.save(output_path, default_blueprint=blueprint)
        print(f"[rerun] No DISPLAY/WAYLAND session detected. Saving recording to: {output_path}")
        return False

    rr.script_setup(args, app_id, default_blueprint=blueprint)
    return True


def log_vector(prefix: str, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    for idx, value in enumerate(flat):
        rr.log(f"{prefix}/d{idx}", rr.Scalars(float(value)))
    rr.log(f"{prefix}_norm", rr.Scalars(float(np.linalg.norm(flat))))


def log_step(
    step_idx: int,
    data: Dict[str, np.ndarray],
    *,
    fps: float,
) -> None:
    rr.set_time("step", sequence=step_idx)
    rr.set_time("seconds", duration=step_idx / fps)

    if "rgb_static" in data:
        rr.log("rgb_static", rr.Image(np.asarray(data["rgb_static"], dtype=np.uint8)))
    if "rgb_gripper" in data:
        rr.log("rgb_gripper", rr.Image(np.asarray(data["rgb_gripper"], dtype=np.uint8)))
    if "robot_obs" in data:
        log_vector("robot_obs", data["robot_obs"])
    if "rel_actions" in data:
        log_vector("rel_actions", data["rel_actions"])
    if "scene_obs" in data:
        log_vector("scene_obs", data["scene_obs"])


def find_episode_index(boundaries: np.ndarray, step_idx: int) -> int:
    matches = np.nonzero((boundaries[:, 0] <= step_idx) & (step_idx < boundaries[:, 1]))[0]
    if len(matches) == 0:
        return -1
    return int(matches[0])


def main() -> None:
    args = parse_args()
    split_dir = args.dataset_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    boundaries = load_boundaries(split_dir)
    step_indices = resolve_step_indices(
        split_dir,
        boundaries,
        episode_index=args.episode_index,
        start_step=args.start_step,
        max_steps=args.max_steps,
        step_stride=args.step_stride,
    )
    if not step_indices:
        raise ValueError("No steps selected for visualization.")

    first_step = load_step_file(step_file(split_dir, step_indices[0]))
    blueprint = setup_blueprint(first_step)
    used_script_setup = configure_rerun(args, blueprint=blueprint)

    episode_annotations = build_episode_annotation_lookup(split_dir, args.lang_folder)

    for step_idx in step_indices:
        if step_idx in episode_annotations:
            annotation = episode_annotations[step_idx]
            rr.set_time("step", sequence=step_idx)
            rr.set_time("seconds", duration=step_idx / args.fps)
            task_prefix = f"[{annotation['task']}] " if annotation["task"] else ""
            rr.log(
                "language",
                rr.TextLog(f"{task_prefix}{annotation['text']}", level=rr.TextLogLevel.INFO),
            )

        episode_idx = find_episode_index(boundaries, step_idx)
        if episode_idx >= 0 and step_idx == int(boundaries[episode_idx][0]):
            episode_start, episode_end = boundaries[episode_idx]
            rr.set_time("step", sequence=step_idx)
            rr.set_time("seconds", duration=step_idx / args.fps)
            rr.log(
                "episode",
                rr.TextLog(
                    f"episode={episode_idx}, start={episode_start}, end={episode_end}, len={episode_end - episode_start}",
                    level=rr.TextLogLevel.INFO,
                ),
            )

        data = load_step_file(step_file(split_dir, step_idx))
        log_step(step_idx, data, fps=args.fps)

    if used_script_setup:
        rr.script_teardown(args)


if __name__ == "__main__":
    main()
