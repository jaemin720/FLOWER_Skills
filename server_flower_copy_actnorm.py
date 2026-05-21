import argparse
import importlib
import json
import logging
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import json_numpy
import numpy as np
import torch
import torchvision
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from omegaconf import OmegaConf

from flower.datasets.utils.episode_utils import (
    load_dataset_statistics,
    process_rgb,
    process_state,
)

json_numpy.patch()

LOGGER = logging.getLogger("server_flower")


def json_response(obj: Any) -> JSONResponse:
    return JSONResponse(json.loads(json_numpy.dumps(obj)))


def npy_response(array: np.ndarray) -> Response:
    buffer = BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


def load_pickle_payload(raw_body: bytes) -> dict:
    payload = pickle.loads(raw_body)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload)!r}")
    return payload


def load_class(name: str):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def find_run_dir(path: Path) -> Path:
    path = path.resolve()
    candidates = [path] if path.is_dir() else list(path.parents)
    for candidate in candidates:
        if (candidate / ".hydra" / "config.yaml").exists():
            return candidate
    raise FileNotFoundError(f"Could not find .hydra/config.yaml above {path}")


def resolve_checkpoint_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {path}")
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    last_ckpt = next(path.rglob("last.ckpt"), None)
    if last_ckpt is not None:
        return last_ckpt

    ckpts = sorted(path.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt file found under {path}")
    return ckpts[-1]


def load_run_config(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def build_val_transforms(cfg, dataset_root: Optional[Path]) -> Dict[str, torchvision.transforms.Compose]:
    transforms_cfg = cfg.datamodule.transforms
    if dataset_root is not None:
        training_dir = dataset_root / "training"
        validation_dir = dataset_root / "validation"
        if training_dir.exists() and validation_dir.exists():
            transforms_cfg = load_dataset_statistics(training_dir, validation_dir, transforms_cfg)
        else:
            LOGGER.warning(
                "Dataset root %s does not contain training/validation folders; "
                "continuing without statistics.yaml overrides.",
                dataset_root,
            )

    val_transforms = {}
    for key in transforms_cfg.val:
        val_transforms[key] = torchvision.transforms.Compose(
            [hydra.utils.instantiate(transform) for transform in transforms_cfg.val[key]]
        )
    return val_transforms


def apply_ema_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ema_weights = checkpoint.get("callbacks", {}).get("EMA", {}).get("ema_weights")
    if ema_weights is None:
        LOGGER.info("No EMA weights found in %s; using checkpoint state_dict.", checkpoint_path)
        return

    model_state = model.state_dict()
    if len(ema_weights) != len(model_state):
        LOGGER.warning(
            "EMA weight count (%d) does not match model state size (%d); "
            "using checkpoint state_dict instead.",
            len(ema_weights),
            len(model_state),
        )
        return

    ema_state = {name: weight for name, weight in zip(model_state.keys(), ema_weights)}
    missing, unexpected = model.load_state_dict(ema_state, strict=False)
    LOGGER.info(
        "Applied EMA weights from %s (missing=%d, unexpected=%d).",
        checkpoint_path,
        len(missing),
        len(unexpected),
    )


def load_model(
    checkpoint_path: Path,
    run_cfg,
    device: torch.device,
    multistep: int,
    action_horizon: int,
    num_sampling_steps: int,
    use_wrist: bool,
    use_torch_compile: bool,
    use_ema_weights: bool,
):
    model_cfg = OmegaConf.to_container(run_cfg.model, resolve=True)
    model_cfg.pop("_recursive_", None)
    class_name = model_cfg.pop("_target_")

    model_cfg["multistep"] = multistep
    model_cfg["act_window_size"] = action_horizon
    model_cfg["num_sampling_steps"] = num_sampling_steps
    model_cfg["use_second_view"] = use_wrist
    model_cfg["second_view_key"] = "rgb_gripper"

    model_class = load_class(class_name)
    model = model_class.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        **model_cfg,
    )

    if use_ema_weights:
        apply_ema_weights(model, checkpoint_path)

    model = model.to(device)
    model.eval()

    if use_torch_compile:
        model = torch.compile(model, mode="default")

    return model


class FlowerInferenceServer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint_path))
        self.run_dir = find_run_dir(self.checkpoint_path)
        self.cfg = load_run_config(self.run_dir)

        cfg_use_wrist = bool(self.cfg.model.get("use_second_view", False))
        self.use_wrist = cfg_use_wrist if args.use_wrist is None else args.use_wrist
        self.use_proprio = bool(self.cfg.model.get("use_proprio", False))
        self.observation_space = self.cfg.datamodule.observation_space
        self.proprio_state = self.cfg.datamodule.proprioception_dims

        dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
        self.transforms = build_val_transforms(self.cfg, dataset_root)
        self.prompt_text: Optional[str] = None

        # 핵심 추가: action denormalization statistics
        self.action_stats = self._load_action_statistics(args.action_stats_path)

        self.model = load_model(
            checkpoint_path=self.checkpoint_path,
            run_cfg=self.cfg,
            device=self.device,
            multistep=args.multistep,
            action_horizon=args.action_horizon,
            num_sampling_steps=args.num_sampling_steps,
            use_wrist=self.use_wrist,
            use_torch_compile=args.use_torch_compile,
            use_ema_weights=not args.no_use_ema,
        )

        LOGGER.info("Loaded checkpoint: %s", self.checkpoint_path)
        LOGGER.info("Run directory: %s", self.run_dir)
        LOGGER.info("Device: %s", self.device)
        LOGGER.info("Using wrist view: %s", self.use_wrist)
        LOGGER.info("Using proprio: %s", self.use_proprio)

        if args.data_name:
            LOGGER.warning("--data_name is ignored by this server.")
        if args.stats_path:
            LOGGER.warning("--stats_path is ignored by this server. Use --action_stats_path instead.")
        if args.image_size is not None:
            LOGGER.warning("--image_size is ignored; image resizing follows Hydra val transforms.")
        if args.control_frequency is not None:
            LOGGER.warning("--control_frequency is ignored by this server.")
        if args.robot_name:
            LOGGER.warning("--robot_name is ignored; prompt formatting comes from the checkpoint config.")
        if args.prompt_style:
            LOGGER.warning("--prompt_style is ignored; prompt formatting comes from the checkpoint config.")
        if args.language_model:
            LOGGER.warning("--language_model is ignored; the checkpoint config defines the VLM.")

    def _load_action_statistics(self, stats_path: Optional[str]) -> Optional[dict]:
        """
        Load action normalization statistics from .npz.

        Supported key patterns:
        - mean/std
        - action_mean/action_std
        - actions_mean/actions_std
        - rel_actions_mean/rel_actions_std
        - min/max
        - action_min/action_max
        - actions_min/actions_max
        - rel_actions_min/rel_actions_max

        Supported denormalization:
        - z-score: raw = normalized * std + mean
        - min-max [-1, 1]: raw = (normalized + 1) / 2 * (max - min) + min
        """
        if not stats_path:
            LOGGER.info("No --action_stats_path provided; model actions will be returned as-is.")
            return None

        stats_file = Path(stats_path).expanduser().resolve()
        if not stats_file.exists():
            raise FileNotFoundError(f"Action statistics file does not exist: {stats_file}")

        data = np.load(stats_file)
        keys = set(data.files)

        LOGGER.info("Loaded action statistics from %s", stats_file)
        LOGGER.info("Action statistics keys: %s", sorted(keys))

        def get_first_available(names):
            for name in names:
                if name in keys:
                    return np.asarray(data[name], dtype=np.float32)
            return None

        mean = get_first_available(
            [
                "action_mean",
                "actions_mean",
                "rel_actions_mean",
                "mean",
            ]
        )
        std = get_first_available(
            [
                "action_std",
                "actions_std",
                "rel_actions_std",
                "std",
            ]
        )

        min_v = get_first_available(
            [
                "action_min",
                "actions_min",
                "rel_actions_min",
                "min",
            ]
        )
        max_v = get_first_available(
            [
                "action_max",
                "actions_max",
                "rel_actions_max",
                "max",
            ]
        )

        if mean is not None and std is not None:
            mean = np.asarray(mean, dtype=np.float32).squeeze()
            std = np.asarray(std, dtype=np.float32).squeeze()

            LOGGER.info("Using z-score action denormalization.")
            LOGGER.info("Action mean shape=%s, std shape=%s", mean.shape, std.shape)

            return {
                "type": "zscore",
                "mean": mean,
                "std": std,
            }

        if min_v is not None and max_v is not None:
            min_v = np.asarray(min_v, dtype=np.float32).squeeze()
            max_v = np.asarray(max_v, dtype=np.float32).squeeze()

            LOGGER.info("Using min-max [-1, 1] action denormalization.")
            LOGGER.info("Action min shape=%s, max shape=%s", min_v.shape, max_v.shape)

            return {
                "type": "minmax",
                "min": min_v,
                "max": max_v,
            }

        raise KeyError(
            f"Could not find supported action statistics keys in {stats_file}. "
            f"Found keys={sorted(keys)}. "
            "Expected mean/std, action_mean/action_std, actions_mean/actions_std, "
            "rel_actions_mean/rel_actions_std, min/max, action_min/action_max, "
            "actions_min/actions_max, or rel_actions_min/rel_actions_max."
        )

    def _align_action_stats(self, stat: np.ndarray, action: np.ndarray, name: str) -> np.ndarray:
        """
        Align statistics shape to action shape.

        Common cases:
        - action: (7,), stat: (7,)
        - action: (H, 7), stat: (7,)
        - action: (B, H, 7), stat: (7,)
        - action: (H, 7), stat: (H, 7)

        If stat is 14D but action is 7D, use the last 7 dims by default.
        This matches the common setup where policy outputs right-arm 7D and
        env action is 14D with left arm padded in front.
        """
        stat = np.asarray(stat, dtype=np.float32).squeeze()
        action = np.asarray(action, dtype=np.float32)

        action_dim = action.shape[-1]

        if stat.ndim == 1:
            if stat.shape[0] == action_dim:
                return stat

            if stat.shape[0] == 14 and action_dim == 7:
                LOGGER.warning(
                    "%s has 14 dims but action has 7 dims; using last 7 dims. "
                    "If your policy controls the first 7 dims, change this slice.",
                    name,
                )
                return stat[7:14]

            raise ValueError(
                f"Cannot align {name} shape {stat.shape} to action shape {action.shape}."
            )

        if stat.shape[-1] == action_dim:
            return stat

        if stat.shape[-1] == 14 and action_dim == 7:
            LOGGER.warning(
                "%s has last dim 14 but action has last dim 7; using last 7 dims.",
                name,
            )
            return stat[..., 7:14]

        raise ValueError(
            f"Cannot align {name} shape {stat.shape} to action shape {action.shape}."
        )

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize model output action.

        action shape can be:
        - (7,)
        - (H, 7)
        - (B, H, 7)
        """
        if self.action_stats is None:
            return action

        action = np.asarray(action, dtype=np.float32)

        if self.action_stats["type"] == "zscore":
            mean = self._align_action_stats(self.action_stats["mean"], action, "action_mean")
            std = self._align_action_stats(self.action_stats["std"], action, "action_std")

            std = np.where(np.abs(std) < 1e-8, 1.0, std).astype(np.float32)
            denorm = action * std + mean

            LOGGER.debug(
                "Denormalized action with z-score: action shape=%s, mean shape=%s, std shape=%s",
                action.shape,
                mean.shape,
                std.shape,
            )
            return denorm.astype(np.float32)

        if self.action_stats["type"] == "minmax":
            min_v = self._align_action_stats(self.action_stats["min"], action, "action_min")
            max_v = self._align_action_stats(self.action_stats["max"], action, "action_max")

            denorm = (action + 1.0) * 0.5 * (max_v - min_v) + min_v

            LOGGER.debug(
                "Denormalized action with min-max: action shape=%s, min shape=%s, max shape=%s",
                action.shape,
                min_v.shape,
                max_v.shape,
            )
            return denorm.astype(np.float32)

        raise ValueError(f"Unknown action stats type: {self.action_stats['type']}")

    def reset(self, text: str):
        self.prompt_text = text
        self.model.reset()

    def _build_observation(self, payload: dict) -> dict:
        image_primary = payload.get("image_primary", payload.get("image", payload.get("rgb_static")))
        image_wrist = payload.get(
            "image_wrist",
            payload.get("wrist_image", payload.get("rgb_gripper")),
        )
        proprio = payload.get("proprio", payload.get("state", payload.get("robot_obs")))

        if image_primary is None:
            raise ValueError("Observation must contain image_primary, image, or rgb_static.")
        if self.use_proprio and proprio is None:
            raise ValueError("Observation must contain proprio, state, or robot_obs.")
        if self.use_wrist and image_wrist is None:
            raise ValueError(
                "Checkpoint expects a wrist view; provide image_wrist, wrist_image, or rgb_gripper."
            )

        rgb_episode = {"rgb_static": np.asarray(image_primary, dtype=np.uint8)}
        if self.use_wrist:
            rgb_episode["rgb_gripper"] = np.asarray(image_wrist, dtype=np.uint8)

        rgb_obs = process_rgb(
            rgb_episode,
            self.observation_space,
            self.transforms,
            device=self.device,
        )["rgb_obs"]

        observation = {
            "rgb_obs": {key: value.to(self.device).unsqueeze(0) for key, value in rgb_obs.items()},
        }

        if self.use_proprio:
            robot_obs = np.asarray(proprio, dtype=np.float32).reshape(1, -1)
            state_obs = process_state(
                {"robot_obs": robot_obs},
                self.observation_space,
                self.transforms,
                self.proprio_state,
            )["robot_obs"]
            observation["robot_obs"] = state_obs.to(self.device).unsqueeze(0)
            observation["robot_obs_raw"] = torch.from_numpy(robot_obs.squeeze(0)).to(self.device)

        return observation

    def infer(self, payload: dict) -> np.ndarray:
        if self.prompt_text is None:
            raise RuntimeError("Prompt is not set. Call /reset first.")

        obs_payload = payload["observation"] if "observation" in payload else payload
        obs = self._build_observation(obs_payload)
        goal = {"lang_text": self.prompt_text}

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode():
            with autocast_context:
                action = self.model.step(obs, goal)

        action_np = action.detach().cpu().numpy().astype(np.float32)

        # 핵심 추가: normalized action -> raw action
        action_np = self.denormalize_action(action_np)

        return np.squeeze(action_np)


def build_app(server: FlowerInferenceServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "checkpoint": str(server.checkpoint_path),
            "run_dir": str(server.run_dir),
            "device": str(server.device),
            "use_wrist": server.use_wrist,
            "use_proprio": server.use_proprio,
            "action_denormalization": server.action_stats is not None,
            "action_denorm_type": None if server.action_stats is None else server.action_stats["type"],
        }

    @app.post("/reset")
    def reset(payload: Dict[str, Any]):
        text = payload.get("text")
        if not text:
            return JSONResponse({"error": "reset payload requires 'text'."}, status_code=400)
        server.reset(text)
        return {"status": "reset", "text": text}

    @app.post("/query")
    def query(payload: Dict[str, Any]):
        try:
            action = server.infer(payload)
            return json_response({"actions": action})
        except Exception as exc:
            LOGGER.exception("Inference failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/query_pickle")
    async def query_pickle(request: Request):
        try:
            payload = load_pickle_payload(await request.body())
            action = server.infer(payload)
            return npy_response(action)
        except Exception as exc:
            LOGGER.exception("Inference failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint_dir",
        dest="checkpoint_path",
        required=True,
        help="Path to a .ckpt file or a run directory containing .hydra and saved_models.",
    )
    parser.add_argument(
        "--dataset_root",
        "--data_path",
        dest="dataset_root",
        default=None,
        help="Optional CALVIN dataset root containing training/ and validation/ folders.",
    )
    parser.add_argument("--data_name", default=None, help="Ignored. Kept for compatibility.")
    parser.add_argument("--stats_path", default=None, help="Ignored. Kept for compatibility.")
    parser.add_argument(
        "--action_stats_path",
        default=None,
        help="Path to action_statistics.npz used to denormalize model actions.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=45587)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=None, help="Ignored. Kept for compatibility.")
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--multistep", type=int, default=1)
    parser.add_argument("--num_sampling_steps", type=int, default=4)
    parser.add_argument("--control_frequency", type=int, default=None, help="Ignored. Kept for compatibility.")
    parser.add_argument("--robot_name", default=None, help="Ignored. Kept for compatibility.")
    parser.add_argument("--prompt_style", default=None, help="Ignored. Kept for compatibility.")
    parser.add_argument("--language_model", default=None, help="Ignored. Kept for compatibility.")

    wrist_group = parser.add_mutually_exclusive_group()
    wrist_group.add_argument(
        "--use_wrist",
        action="store_true",
        dest="use_wrist",
        help="Force-enable wrist image input.",
    )
    wrist_group.add_argument(
        "--no_use_wrist",
        action="store_false",
        dest="use_wrist",
        help="Force-disable wrist image input.",
    )
    parser.set_defaults(use_wrist=None)

    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--no_use_ema", action="store_true", help="Do not replace checkpoint weights with EMA.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    args = parse_args()
    server = FlowerInferenceServer(args)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()