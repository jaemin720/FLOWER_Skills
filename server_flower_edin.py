import argparse
import contextlib
import io
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import safetensors
import tensorflow as tf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from flower_vla.agents.lang_encoders.florence_tokens import TokenVLM
from flower_vla.dataset.dataset import make_dataset_from_rlds
from flower_vla.dataset.oxe import make_oxe_dataset_kwargs_and_weights
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.dataset.utils.data_utils import NormalizationType

LOGGER = logging.getLogger("server_flower")


def load_state_dict_with_compat(agent: torch.nn.Module, checkpoint_path: Path):
    state_dict = safetensors.safe_open(str(checkpoint_path), framework="pt")
    renamed = {}
    for key in state_dict.keys():
        new_key = (
            key.replace("c_fc1", "fc1")
            .replace("c_fc2", "fc2")
            .replace("c_proj", "proj")
        )
        renamed[new_key] = state_dict.get_tensor(key)
    missing, unexpected = agent.load_state_dict(renamed, strict=False)
    return missing, unexpected


def resize_uint8_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image = tf.image.resize(
        image,
        size=(size, size),
        method="lanczos3",
        antialias=True,
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


def normalize_bounds(values: np.ndarray, p01: np.ndarray, p99: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.clip(
        2.0 * (values - p01) / (p99 - p01 + 1e-8) - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)


def denormalize_action_bounds(
    action: np.ndarray, action_p01: np.ndarray, action_p99: np.ndarray
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    out = action.copy()
    if out.shape[-1] < 7:
        raise ValueError(f"Expected action last dim >= 7, got {out.shape}")

    scaled = (
        (out[..., :6] + 1.0)
        * 0.5
        * (action_p99[:6] - action_p01[:6])
        + action_p01[:6]
    )
    out[..., :6] = scaled
    return out.astype(np.float32)


def to_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def load_dataset_statistics(
    data_name: str,
    data_path: str,
    load_camera_views: list[str],
    stats_path: Optional[str] = None,
) -> dict:
    if stats_path:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        LOGGER.info("Loaded dataset statistics from %s", stats_path)
        return stats

    kwargs_list, _ = make_oxe_dataset_kwargs_and_weights(
        data_name,
        data_path,
        load_camera_views=load_camera_views,
        action_proprio_normalization_type=NormalizationType.BOUNDS,
    )
    if len(kwargs_list) != 1:
        raise ValueError(
            f"Expected exactly one dataset for inference stats, got {len(kwargs_list)}"
        )
    _, stats = make_dataset_from_rlds(**kwargs_list[0], train=True)
    LOGGER.info("Loaded dataset statistics from TFDS cache for %s", data_name)
    return stats


class FlowerInferenceServer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(args.checkpoint_dir).resolve()
        self.run_dir = self.checkpoint_dir.parent
        self.use_wrist = args.use_wrist
        self.image_size = args.image_size
        self.frequency = torch.tensor([args.control_frequency], dtype=torch.int64)
        self.action_space_index = torch.tensor(
            [get_action_space_index("EEF_POS", 1, "velocity", return_tensor=False)],
            dtype=torch.int64,
        )
        self.format_instruction = lambda instruction: generate_policy_prompt(
            instruction=instruction,
            robot_name=args.robot_name,
            num_arms=1,
            action_space="Delta End-Effector",
            prompt_style=args.prompt_style,
        )
        self.tokenizer = TokenVLM(args.language_model)
        self.stats = load_dataset_statistics(
            data_name=args.data_name,
            data_path=args.data_path,
            load_camera_views=["primary", "wrist"] if self.use_wrist else ["primary"],
            stats_path=args.stats_path,
        )
        self.action_p01 = np.asarray(self.stats["action"]["p01"], dtype=np.float32)
        self.action_p99 = np.asarray(self.stats["action"]["p99"], dtype=np.float32)
        self.proprio_p01 = np.asarray(self.stats["proprio"]["p01"], dtype=np.float32)
        self.proprio_p99 = np.asarray(self.stats["proprio"]["p99"], dtype=np.float32)
        self.proprio_dim = int(self.proprio_p01.shape[0])
        self.action_dim = int(self.action_p01.shape[0])
        self.action_horizon = int(args.action_horizon)
        self._lock = threading.Lock()
        self.prompt_text: Optional[str] = None

        self.agent = self._load_agent(
            multistep=args.multistep,
            action_horizon=args.action_horizon,
            num_sampling_steps=args.num_sampling_steps,
            use_torch_compile=args.use_torch_compile,
        )

    def _load_agent(
        self,
        multistep: int,
        action_horizon: int,
        num_sampling_steps: int,
        use_torch_compile: bool,
    ):
        hydra_dir = self.run_dir / ".hydra"
        if not hydra_dir.exists():
            raise FileNotFoundError(f"Missing Hydra config directory: {hydra_dir}")

        with initialize_config_dir(config_dir=str(hydra_dir)):
            cfg = compose(config_name="config")

        with open_dict(cfg):
            cfg.batch_size = 1

        with open_dict(cfg.trainer.agent.agent):
            cfg.trainer.agent.agent.use_proprio = True
            cfg.trainer.agent.agent.return_act_chunk = True
            cfg.trainer.agent.agent.multistep = multistep
            cfg.trainer.agent.agent.act_window_size = action_horizon
            cfg.trainer.agent.agent.num_sampling_steps = num_sampling_steps
            cfg.trainer.agent.agent.use_second_view = bool(self.use_wrist)
            cfg.trainer.agent.agent.second_view_key = "image_wrist"

        agent = hydra.utils.instantiate(
            cfg.trainer.agent,
            device=self.device,
            process_id=0,
        )

        safetensors_path = self.checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Missing checkpoint weights: {safetensors_path}")

        missing, unexpected = load_state_dict_with_compat(agent, safetensors_path)
        LOGGER.info("Loaded checkpoint: %s", safetensors_path)
        LOGGER.info("Missing keys: %d, unexpected keys: %d", len(missing), len(unexpected))

        agent.to(dtype=torch.bfloat16)
        agent.eval()

        if use_torch_compile:
            agent.agent = torch.compile(agent.agent, mode="default")

        return agent

    def reset(self, text: str):
        with self._lock:
            self.prompt_text = text
            self.agent.agent.reset()

    def _build_task(self) -> dict:
        if self.prompt_text is None:
            raise RuntimeError("Prompt is not set. Call /reset first.")

        formatted = self.format_instruction(self.prompt_text)
        language_tokens = self.tokenizer([formatted])
        return {
            "language_instruction": language_tokens,
            "frequency": self.frequency,
            "action_space_index": self.action_space_index,
        }

    def _build_observation(self, payload: dict) -> dict:
        image_primary = payload.get("image_primary", payload.get("image"))
        image_wrist = payload.get("image_wrist", payload.get("wrist_image"))
        proprio = payload.get("proprio", payload.get("state"))

        if image_primary is None:
            raise ValueError("Observation must contain image_primary or image.")
        if proprio is None:
            raise ValueError("Observation must contain proprio or state.")

        image_primary = resize_uint8_image(np.asarray(image_primary, dtype=np.uint8), self.image_size)
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if proprio.shape[0] != self.proprio_p01.shape[0]:
            raise ValueError(
                f"Expected proprio dim {self.proprio_p01.shape[0]}, got {proprio.shape[0]}"
            )
        proprio = normalize_bounds(proprio, self.proprio_p01, self.proprio_p99)

        observation = {
            "image_primary": torch.from_numpy(image_primary).unsqueeze(0).unsqueeze(0),
            "proprio": torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0),
            "pad_mask_dict": {
                "image_primary": torch.ones(1, 1, dtype=torch.bool),
            },
        }

        if self.use_wrist:
            if image_wrist is None:
                raise ValueError("Server is configured with --use_wrist but observation has no wrist image.")
            image_wrist = resize_uint8_image(np.asarray(image_wrist, dtype=np.uint8), self.image_size)
            observation["image_wrist"] = torch.from_numpy(image_wrist).unsqueeze(0).unsqueeze(0)
            observation["pad_mask_dict"]["image_wrist"] = torch.ones(1, 1, dtype=torch.bool)

        return observation

    def infer(self, payload: dict) -> np.ndarray:
        with self._lock:
            obs_payload = payload["observation"] if "observation" in payload else payload
            batch = {
                "observation": self._build_observation(obs_payload),
                "task": self._build_task(),
            }

            autocast_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else contextlib.nullcontext()
            )
            with torch.no_grad():
                with autocast_context:
                    actions = self.agent(batch).detach().cpu().numpy()

        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 2 and actions.shape[0] == 1:
            actions = actions[0]

        actions = denormalize_action_bounds(actions, self.action_p01, self.action_p99)
        return actions.astype(np.float32)


def build_app(server: FlowerInferenceServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "device": server.device.type,
            "image_size": server.image_size,
            "use_wrist": server.use_wrist,
            "proprio_dim": server.proprio_dim,
            "action_dim": server.action_dim,
            "action_horizon": server.action_horizon,
            "supports": {
                "json": True,
                "pickle": True,
                "npy_response": True,
            },
            "recommended_transport": "pickle",
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
            actions = server.infer(payload)
            return {"actions": actions.tolist()}
        except Exception as exc:
            LOGGER.exception("Inference failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/query_pickle")
    async def query_pickle(request: Request):
        body = await request.body()
        if not body:
            return JSONResponse({"error": "Empty request body."}, status_code=400)

        try:
            payload = pickle.loads(body)
        except Exception as exc:
            return JSONResponse({"error": f"Failed to decode pickle payload: {exc}"}, status_code=400)

        try:
            actions = server.infer(payload)
            return Response(
                content=to_npy_bytes(actions),
                media_type="application/octet-stream",
            )
        except Exception as exc:
            LOGGER.exception("Inference failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Path to FLOWER checkpoint directory, e.g. .../checkpoint_5000")
    parser.add_argument("--data_name", required=True, help="Dataset name used for training, e.g. my_robot_right_full_state or agi_multitask")
    parser.add_argument("--data_path", required=True, help="TFDS root path, e.g. /home/jack/tensorflow_datasets")
    parser.add_argument("--stats_path", default=None, help="Optional explicit dataset statistics JSON path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=45587)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--multistep", type=int, default=1)
    parser.add_argument("--num_sampling_steps", type=int, default=4)
    parser.add_argument("--control_frequency", type=int, default=10)
    parser.add_argument("--robot_name", default="CustomRobot")
    parser.add_argument("--prompt_style", default="minimal")
    parser.add_argument("--language_model", default="microsoft/Florence-2-large")
    parser.add_argument("--use_wrist", action="store_true", help="Enable wrist image as second view. Only use this if the checkpoint was trained with use_second_view=True.")
    parser.add_argument("--use_torch_compile", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    server = FlowerInferenceServer(args)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
