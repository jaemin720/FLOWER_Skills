from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute QueST SkillVAE labels for a FLOWER disk dataset.")
    parser.add_argument("--dataset_dir", required=True, type=Path, help="FLOWER split directory, e.g. .../training")
    parser.add_argument("--quest_checkpoint", required=True, type=Path, help="QueST stage-0 checkpoint or directory")
    parser.add_argument("--quest_repo_path", type=Path, default=Path("/home/jack/quest_practice/QueST"))
    parser.add_argument("--output", type=Path, default=None, help="Output .npy path. Defaults under dataset_dir.")
    parser.add_argument("--key", choices=("lang", "vision"), default="lang", help="FLOWER dataset key to mirror.")
    parser.add_argument("--lang_folder", default="lang_clip_resnet50")
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--aux_lang_loss_window", type=int, default=8)
    parser.add_argument("--action_key", default="rel_actions")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=512)
    parser.add_argument("--vq_type", choices=("fsq", "vq"), default="fsq")
    parser.add_argument("--fsq_level", nargs="*", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def get_default_fsq_level(codebook_size: int) -> List[int]:
    power = int(math.log2(codebook_size))
    fsq_levels = {
        4: [5, 3],
        6: [8, 8],
        8: [8, 6, 5],
        9: [8, 8, 8],
        10: [8, 5, 5, 5],
        11: [8, 8, 6, 5],
        12: [7, 5, 5, 5, 5],
    }
    if power not in fsq_levels:
        raise ValueError(f"No default FSQ level for codebook_size={codebook_size}")
    return fsq_levels[power]


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"QueST checkpoint path does not exist: {path}")

    candidates = []
    for suffix in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(path.glob(suffix))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in QueST checkpoint directory: {path}")
    return sorted(candidates, key=lambda item: item.stat().st_mtime)[-1]


def build_action_cache(dataset_dir: Path, action_key: str) -> Path:
    cache_path = dataset_dir / f"quest_{action_key}.npy"
    if cache_path.is_file():
        return cache_path

    ep_start_end_ids = np.load(dataset_dir / "ep_start_end_ids.npy").astype(np.int64)
    total_steps = int(ep_start_end_ids[:, 1].max())
    with np.load(dataset_dir / "episode_0000000.npz") as first:
        first_action = np.asarray(first[action_key], dtype=np.float32)

    actions = np.empty((total_steps, first_action.shape[0]), dtype=np.float32)
    actions[0] = first_action
    for step_idx in tqdm(range(1, total_steps), desc=f"Caching {action_key}"):
        with np.load(dataset_dir / f"episode_{step_idx:07d}.npz") as data:
            actions[step_idx] = np.asarray(data[action_key], dtype=np.float32)
    np.save(cache_path, actions)
    return cache_path


def load_lang_indices(dataset_dir: Path, lang_folder: str) -> np.ndarray:
    lang_path = dataset_dir / lang_folder / "auto_lang_ann.npy"
    if not lang_path.is_file():
        lang_path = dataset_dir / "auto_lang_ann.npy"
    lang_data = np.load(lang_path, allow_pickle=True).item()
    return np.asarray(lang_data["info"]["indx"], dtype=np.int64)


def build_start_indices(args: argparse.Namespace, dataset_dir: Path) -> np.ndarray:
    seq_len = args.seq_len
    if args.key == "lang":
        ep_start_end_ids = load_lang_indices(dataset_dir, args.lang_folder)
    else:
        ep_start_end_ids = np.load(dataset_dir / "ep_start_end_ids.npy").astype(np.int64)

    starts = []
    for start, end in ep_start_end_ids:
        start = int(start)
        end = int(end)
        if args.key == "lang" and args.pretrain:
            start = max(start, end + 1 - seq_len - args.aux_lang_loss_window)
        count = 0
        for start_idx in range(start, end + 1 - seq_len):
            if args.key != "lang" or count % args.skip_frames == 0:
                starts.append(start_idx)
            count += 1
    return np.asarray(starts, dtype=np.int64)


def load_autoencoder(args: argparse.Namespace):
    quest_repo_path = str(args.quest_repo_path.expanduser())
    if quest_repo_path not in sys.path:
        sys.path.insert(0, quest_repo_path)

    from quest.algos.quest_modules.skill_vae import SkillVAE

    fsq_level: Optional[List[int]] = args.fsq_level
    if args.vq_type == "fsq" and fsq_level is None:
        fsq_level = get_default_fsq_level(args.codebook_size)

    autoencoder = SkillVAE(
        action_dim=args.action_dim,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        skill_block_size=args.seq_len,
        downsample_factor=args.downsample_factor,
        attn_pdrop=0.1,
        use_causal_encoder=True,
        use_causal_decoder=True,
        encoder_heads=4,
        encoder_layers=2,
        decoder_heads=4,
        decoder_layers=4,
        vq_type=args.vq_type,
        fsq_level=fsq_level,
        codebook_dim=args.codebook_dim,
        codebook_size=args.codebook_size,
    )

    checkpoint_path = resolve_checkpoint(args.quest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model_state = autoencoder.state_dict()
    autoencoder_state = {}
    prefixes = ("autoencoder.", "policy.autoencoder.", "agent.autoencoder.")
    for key, value in state_dict.items():
        stripped_key = key
        for prefix in prefixes:
            if stripped_key.startswith(prefix):
                stripped_key = stripped_key[len(prefix):]
                break
        if stripped_key in model_state and model_state[stripped_key].shape == value.shape:
            autoencoder_state[stripped_key] = value
    if not autoencoder_state:
        raise RuntimeError(f"No matching SkillVAE tensors found in checkpoint: {checkpoint_path}")
    autoencoder.load_state_dict(autoencoder_state, strict=False)
    print(f"Loaded QueST SkillVAE checkpoint: {len(autoencoder_state)} tensors <- {checkpoint_path}")
    return autoencoder


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    output_path = args.output
    if output_path is None:
        output_path = dataset_dir / f"quest_skill_indices_s{args.seq_len}_d{args.downsample_factor}.npy"
    output_path = output_path.expanduser()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    action_cache = build_action_cache(dataset_dir, args.action_key)
    actions = np.load(action_cache, mmap_mode="r")
    start_indices = build_start_indices(args, dataset_dir)
    autoencoder = load_autoencoder(args).to(device).eval()

    skill_num_tokens = args.seq_len // args.downsample_factor
    labels = np.empty((len(start_indices), skill_num_tokens), dtype=np.int64)

    for offset in tqdm(range(0, len(start_indices), args.batch_size), desc="Encoding QueST skills"):
        batch_starts = start_indices[offset: offset + args.batch_size]
        batch_actions = np.stack([actions[start:start + args.seq_len] for start in batch_starts], axis=0)
        batch_tensor = torch.as_tensor(batch_actions, device=device, dtype=torch.float32)
        batch_labels = autoencoder.get_indices(batch_tensor).long().cpu().numpy()
        labels[offset: offset + len(batch_starts)] = batch_labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)
    print(f"Saved QueST skill labels: {labels.shape} -> {output_path}")


if __name__ == "__main__":
    main()
