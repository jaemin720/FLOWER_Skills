"""
Inspect whether actions saved in a converted FLOWER dataset match the original
raw pickle data and whether the action distribution is overly close to zero.

Typical usage:

python preprocess/inspect_converted_actions.py \
  --raw_input /path/to/raw_task \
  --converted_root /path/to/converted_dataset \
  --split training \
  --action_source state_delta \
  --static_cam_key front_cam \
  --gripper_cam_key right_wrist_cam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.convert_raw_pkl_to_flower import (  # noqa: E402
    EpisodeRecord,
    build_rel_action,
    build_robot_obs,
    load_episode_records,
    parse_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_input", required=True, type=Path, help="Raw pickle file or directory used for conversion.")
    parser.add_argument("--converted_root", required=True, type=Path, help="Converted dataset root containing training/validation.")
    parser.add_argument("--split", choices=["training", "validation", "both"], default="training")
    parser.add_argument("--pkl_glob", default="**/*.pkl", help="Glob pattern for raw pickle discovery.")
    parser.add_argument("--action_source", choices=["raw", "state_delta"], default="raw")
    parser.add_argument("--prefer_intervene_action", action="store_true")
    parser.add_argument("--action_slice", default="7:14")
    parser.add_argument("--obs_state_key", default="state")
    parser.add_argument("--state_pose_slice", default="23:29")
    parser.add_argument("--state_gripper_index", type=int, default=19)
    parser.add_argument("--include_right_ft", action="store_true")
    parser.add_argument("--state_force_slice", default="20:23")
    parser.add_argument("--state_torque_slice", default="29:32")
    parser.add_argument("--sample_count", type=int, default=5, help="How many mismatching examples to print.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for exact action equality checks.")
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> Dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def action_norm_stats(actions: np.ndarray) -> Dict[str, float]:
    if len(actions) == 0:
        return {}
    pos_norm = np.linalg.norm(actions[:, :3], axis=1)
    rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
    full_norm = np.linalg.norm(actions[:, :6], axis=1)
    return {
        "pos_lt_1e-4": float(np.mean(pos_norm < 1e-4)),
        "pos_lt_1e-3": float(np.mean(pos_norm < 1e-3)),
        "pos_lt_1e-2": float(np.mean(pos_norm < 1e-2)),
        "rot_lt_1e-4": float(np.mean(rot_norm < 1e-4)),
        "rot_lt_1e-3": float(np.mean(rot_norm < 1e-3)),
        "rot_lt_1e-2": float(np.mean(rot_norm < 1e-2)),
        "full_lt_1e-4": float(np.mean(full_norm < 1e-4)),
        "full_lt_1e-3": float(np.mean(full_norm < 1e-3)),
        "full_lt_1e-2": float(np.mean(full_norm < 1e-2)),
    }


def summarize_array(name: str, values: np.ndarray) -> None:
    print(f"\n[{name}]")
    print("shape:", values.shape)
    print("mean:", np.mean(values, axis=0))
    print("std :", np.std(values, axis=0))
    print("min :", np.min(values, axis=0))
    print("max :", np.max(values, axis=0))
    print("near-zero ratios:", action_norm_stats(values))


def inspect_split(
    split_dir: Path,
    raw_lookup: Dict[Tuple[str, int], EpisodeRecord],
    *,
    action_source: str,
    obs_state_key: str,
    pose_slice: slice,
    gripper_index: int,
    include_right_ft: bool,
    force_slice: slice,
    torque_slice: slice,
    action_slice: slice,
    prefer_intervene_action: bool,
    atol: float,
    sample_count: int,
) -> None:
    metadata = load_metadata(split_dir / "conversion_metadata.json")
    mismatch_examples: List[Dict] = []
    action_diffs = []
    robot_obs_diffs = []
    saved_actions = []
    expected_actions = []
    raw_right_actions = []
    saved_robot_obs = []
    expected_robot_obs = []

    for episode_info in metadata["metadata"]:
        key = (episode_info["source_name"], int(episode_info["source_episode_index"]))
        if key not in raw_lookup:
            raise KeyError(f"Could not find raw episode for metadata key {key}")

        raw_record = raw_lookup[key]
        start = int(episode_info["start_step_index"])
        end = int(episode_info["end_step_index"])
        num_steps = end - start
        if num_steps != len(raw_record.steps):
            raise ValueError(
                f"Step count mismatch for {key}: metadata says {num_steps}, raw episode has {len(raw_record.steps)}"
            )

        for offset, raw_step in enumerate(raw_record.steps):
            saved_step_path = split_dir / f"episode_{start + offset:07d}.npz"
            saved_step = np.load(saved_step_path)

            saved_action = saved_step["rel_actions"].astype(np.float32)
            expected_action = build_rel_action(
                raw_step,
                action_source=action_source,
                state_key=obs_state_key,
                pose_slice=pose_slice,
                action_slice=action_slice,
                prefer_intervene_action=prefer_intervene_action,
            ).astype(np.float32)

            saved_ro = saved_step["robot_obs"].astype(np.float32)
            expected_ro = build_robot_obs(
                raw_step,
                obs_state_key,
                pose_slice,
                gripper_index,
                include_right_ft=include_right_ft,
                force_slice=force_slice,
                torque_slice=torque_slice,
            ).astype(np.float32)

            raw_right_action = np.asarray(raw_step["actions"]).squeeze()[action_slice].astype(np.float32)

            saved_actions.append(saved_action)
            expected_actions.append(expected_action)
            raw_right_actions.append(raw_right_action)
            saved_robot_obs.append(saved_ro)
            expected_robot_obs.append(expected_ro)

            action_diff = saved_action - expected_action
            robot_obs_diff = saved_ro - expected_ro
            action_diffs.append(action_diff)
            robot_obs_diffs.append(robot_obs_diff)

            if len(mismatch_examples) < sample_count and not np.allclose(saved_action, expected_action, atol=atol):
                mismatch_examples.append(
                    {
                        "source": key,
                        "step_offset": offset,
                        "saved_action": saved_action,
                        "expected_action": expected_action,
                        "diff": action_diff,
                    }
                )

    saved_actions_np = np.stack(saved_actions, axis=0)
    expected_actions_np = np.stack(expected_actions, axis=0)
    raw_right_actions_np = np.stack(raw_right_actions, axis=0)
    saved_robot_obs_np = np.stack(saved_robot_obs, axis=0)
    expected_robot_obs_np = np.stack(expected_robot_obs, axis=0)
    action_diffs_np = np.stack(action_diffs, axis=0)
    robot_obs_diffs_np = np.stack(robot_obs_diffs, axis=0)

    print(f"\n=== Split: {split_dir.name} ===")
    print("episodes:", len(metadata["metadata"]))
    print("steps:", len(saved_actions_np))
    print("action exact-match ratio:", float(np.mean(np.all(np.isclose(saved_actions_np, expected_actions_np, atol=atol), axis=1))))
    print("action mean abs diff:", np.mean(np.abs(action_diffs_np), axis=0))
    print("action max abs diff :", np.max(np.abs(action_diffs_np), axis=0))
    print("robot_obs mean abs diff:", np.mean(np.abs(robot_obs_diffs_np), axis=0))
    print("robot_obs max abs diff :", np.max(np.abs(robot_obs_diffs_np), axis=0))

    summarize_array("saved_actions", saved_actions_np)
    summarize_array("expected_actions_from_raw", expected_actions_np)
    summarize_array("raw_right_actions", raw_right_actions_np)

    print("\n[saved_robot_obs]")
    print("shape:", saved_robot_obs_np.shape)
    print("mean:", np.mean(saved_robot_obs_np, axis=0))
    print("std :", np.std(saved_robot_obs_np, axis=0))

    if mismatch_examples:
        print("\n[mismatch_examples]")
        for example in mismatch_examples:
            print(example)
    else:
        print("\nNo action mismatches found within atol =", atol)


def main() -> None:
    args = parse_args()
    pose_slice = parse_slice(args.state_pose_slice)
    force_slice = parse_slice(args.state_force_slice)
    torque_slice = parse_slice(args.state_torque_slice)
    action_slice = parse_slice(args.action_slice)

    raw_records = load_episode_records(args.raw_input, args.pkl_glob)
    raw_lookup = {(record.source_name, record.source_episode_index): record for record in raw_records}
    print("raw episodes:", len(raw_records))

    splits = ["training", "validation"] if args.split == "both" else [args.split]
    for split in splits:
        inspect_split(
            args.converted_root / split,
            raw_lookup,
            action_source=args.action_source,
            obs_state_key=args.obs_state_key,
            pose_slice=pose_slice,
            gripper_index=args.state_gripper_index,
            include_right_ft=args.include_right_ft,
            force_slice=force_slice,
            torque_slice=torque_slice,
            action_slice=action_slice,
            prefer_intervene_action=args.prefer_intervene_action,
            atol=args.atol,
            sample_count=args.sample_count,
        )


if __name__ == "__main__":
    main()
