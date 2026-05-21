import argparse
import numpy as np
import agentlace.inference as ali
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from concurrent.futures import ThreadPoolExecutor
import pickle as pkl
import datetime, copy
from pynput import keyboard
import os
from collections import deque

# 이 파일: Real_Robo/forcevla/openpi-client/src/openpi_client/...
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]  # => Real_Robo
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.config import load_config, build_environment
global step_count
step_count = 0
global step_count_list
step_count_list = []
skip_key = False
def on_press(key):
    global skip_key
    global save_key
    try:
        if key == keyboard.Key.f10:
            skip_key = True
            step_count_list.append(step_count)
            print("Step count list : ", step_count_list)
    except AttributeError:
        pass

listener = keyboard.Listener(
    on_press=on_press)
listener.start()


POLICY_DIM = 7
ENV_DIM = 14

global state_history_queue
state_history_queue = None



global episode_force_max_list
episode_force_max_list = []

global current_episode_force_max
current_episode_force_max = None

FORCE_INDICES = [20, 21, 22]

def extract_force_from_obs(obs):
    """
    obs에서 force 관련 값만 꺼냄.
    obs["state"] 안에 force가 있다고 가정.
    """
    if not isinstance(obs, dict):
        return None
    if "state" not in obs:
        return None

    raw_state = np.asarray(obs["state"]).squeeze().astype(np.float32)

    if raw_state.ndim != 1:
        raw_state = raw_state.reshape(-1)

    if max(FORCE_INDICES) >= len(raw_state):
        return None

    return raw_state[FORCE_INDICES]


def update_episode_force_max(obs):
    """
    현재 obs의 force를 보고 episode 내 최대값 갱신.
    음수/양수 모두 고려하려면 abs 기준으로 보는 게 보통 안전함.
    """
    global current_episode_force_max

    force = extract_force_from_obs(obs)
    if force is None:
        return

    # 축별 값 중 절댓값 기준 최대
    # 예: [-3, 2, -8] -> 8
    cur_max = float(np.max(np.abs(force)))

    if current_episode_force_max is None:
        current_episode_force_max = cur_max
    else:
        current_episode_force_max = max(current_episode_force_max, cur_max)


def pad_action_to_env(x: np.ndarray) -> np.ndarray:
    """
    policy action (..., 7) -> env action (..., 14) by zero-padding FRONT 7 dims.
    also handles cases like (7, H) by transposing to (H, 7).
    """
    x = np.asarray(x, dtype=np.float32)

    # handle (7, H) style
    if x.ndim == 2 and x.shape[-1] != POLICY_DIM and x.shape[0] == POLICY_DIM:
        x = x.T  # -> (H, 7)

    # already env dim
    if x.shape[-1] == ENV_DIM:
        return x

    if x.shape[-1] != POLICY_DIM:
        raise ValueError(f"Unexpected action shape {x.shape}; expected last dim {POLICY_DIM} or {ENV_DIM}")

    pad = np.zeros((*x.shape[:-1], ENV_DIM - POLICY_DIM), dtype=np.float32)  # (..., 7)
    return np.concatenate([pad, x], axis=-1)  # (..., 14)



def save_traj(transitions, success_needed, _name, _task):

    _date = datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(f"./online_demos/{_name}"):
        os.makedirs(f"./online_demos/{_name}")
        
    if not os.path.exists(f"./online_demos/{_name}/{_task}"):
        os.makedirs(f"./online_demos/{_name}/{_task}")
    if not os.path.exists(f"./online_demos/{_name}/{_task}/{_date}"):
        os.makedirs(f"./online_demos/{_name}/{_task}/{_date}")

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./online_demos/{_name}/{_task}/{_date}/{_task}_{success_needed}_online_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")


# def npify_obs(obs):
#     if isinstance(obs, dict):
#         return {k: np.asarray(v) for k, v in obs.items()}
#     return np.asarray(obs)
process_keys = {
    "state": "state",
    # "left_wrist_cam": "image",
    "front_cam": "image",
    "right_wrist_cam": "wrist_image",
    # "right_force_history": "force_history"
}

# def npify_obs(obs, residual=False):
#     global state_history_queue
#     if not isinstance(obs, dict):
#         return np.asarray(obs)
#     new_obs = {}
#     for local_key, server_key in process_keys.items():
#         if local_key in obs:
#             if not residual :
#                 if local_key == "right_force_history":
#                     new_obs[server_key] = np.asarray(obs[local_key]).squeeze(0).reshape(-1)
#                 else:
#                     new_obs[server_key] = np.asarray(obs[local_key]).squeeze(0)
#             else: 
#                 raw_data = np.asarray(obs[local_key]).squeeze()
#                 new_obs[server_key] = np.stack([raw_data, raw_data], axis=0)

#     if not residual:
#         raw_state = np.array(new_obs['state']).squeeze().astype(np.float32)
#     else: 
#         raw_state_single = np.asarray(obs["state"]).squeeze().astype(np.float32)
#     target_indices = [23,24,25,26,27,28, 19, 20,21,22, 29,30,31]
#     # target_indices = [23,24,25,26,27,28, 19]

#     if not residual:
#         current_state_feature = raw_state[target_indices]
#     else :
#         current_state_feature = raw_state_single[target_indices]

#     if ft_stack > 0:
#         if state_history_queue is None:
#             state_history_queue = deque(maxlen=ft_stack)
#         if len(state_history_queue) == 0:
#             for _ in range(ft_stack):
#                 state_history_queue.append(current_state_feature)
#         else:
#             state_history_queue.append(current_state_feature)
#         stacked_state = np.stack(state_history_queue, axis=0)
#         new_obs['state'] = stacked_state
#     else:
#         if not residual:
#             new_obs['state'] = current_state_feature
#         else : 
#             new_obs['state'] = np.stack([current_state_feature, current_state_feature], axis=0)
#     # new_obs["state"] = raw_state[target_indices]
    
#     new_obs["prompt"] = prompt
#     return new_obs

# 데이터 전처리와 동일하게 맞춘 코드
def npify_obs(obs, residual=False):
    global state_history_queue
    if not isinstance(obs, dict):
        return np.asarray(obs)
    new_obs = {}
    for local_key, server_key in process_keys.items():
        if local_key in obs:
            if not residual :
                if local_key == "right_force_history":
                    new_obs[server_key] = np.asarray(obs[local_key]).squeeze(0).reshape(-1)
                else:
                    new_obs[server_key] = np.asarray(obs[local_key]).squeeze(0)
            else: 
                raw_data = np.asarray(obs[local_key]).squeeze()
                new_obs[server_key] = np.stack([raw_data, raw_data], axis=0)

    if not residual:
        raw_state = np.array(new_obs['state']).squeeze().astype(np.float32)
    else: 
        raw_state_single = np.asarray(obs["state"]).squeeze().astype(np.float32)
        
    target_indices = [23,24,25,26,27,28, 19, 20,21,22, 29,30,31]

    if not residual:
        current_state_feature = raw_state[target_indices]
    else :
        current_state_feature = raw_state_single[target_indices]

    # 평가 시에도 정확한 최신 F/T 센서 값을 history에서 가져와 교체해야 합니다.
    if "right_force_history" in obs:
        right_force_history = np.asarray(obs['right_force_history']).squeeze()
        # 형태가 (10, 6) 또는 1차원 배열일 수 있으므로 마지막 스텝(-1)의 6차원을 가져옵니다.
        # 차원 축소 후 마지막 6개 값을 가져오도록 안전하게 처리
        current_right_force = right_force_history.reshape(-1, 6)[-1] 
        
        current_state_feature[7:10] = current_right_force[0:3]  # Force 교체
        current_state_feature[10:13] = current_right_force[3:6] # Torque 교체
    # ----------------------------------------

    if ft_stack > 0:
        if state_history_queue is None:
            state_history_queue = deque(maxlen=ft_stack)
        if len(state_history_queue) == 0:
            for _ in range(ft_stack):
                state_history_queue.append(current_state_feature)
        else:
            state_history_queue.append(current_state_feature)
        stacked_state = np.stack(state_history_queue, axis=0)
        new_obs['state'] = stacked_state
    else:
        if not residual:
            new_obs['state'] = current_state_feature
        else : 
            new_obs['state'] = np.stack([current_state_feature, current_state_feature], axis=0)
    
    new_obs["prompt"] = prompt
    return new_obs
    if "right_force_history" in obs:
        right_force_history = np.asarray(obs['right_force_history']).squeeze()
        # 형태가 (10, 6) 또는 1차원 배열일 수 있으므로 마지막 스텝(-1)의 6차원을 가져옵니다.
        # 차원 축소 후 마지막 6개 값을 가져오도록 안전하게 처리
        current_right_force = right_force_history.reshape(-1, 6)[-1] 
        
        current_state_feature[7:10] = current_right_force[0:3]  # Force 교체
        current_state_feature[10:13] = current_right_force[3:6] # Torque 교체
    # ----------------------------------------

    if ft_stack > 0:
        if state_history_queue is None:
            state_history_queue = deque(maxlen=ft_stack)
        if len(state_history_queue) == 0:
            for _ in range(ft_stack):
                state_history_queue.append(current_state_feature)
        else:
            state_history_queue.append(current_state_feature)
        stacked_state = np.stack(state_history_queue, axis=0)
        new_obs['state'] = stacked_state
    else:
        if not residual:
            new_obs['state'] = current_state_feature
        else : 
            new_obs['state'] = np.stack([current_state_feature, current_state_feature], axis=0)
    
    new_obs["prompt"] = prompt
    return new_obs

def main():
    global skip_key
    global step_count
    global step_count_list
    p = argparse.ArgumentParser()
    p.add_argument("--trainer_ip", type=str, default="127.0.0.1")
    p.add_argument("--trainer_port", type=int, default=45587) 
    p.add_argument("--save_video", action="store_true", default=False)
    p.add_argument("--classifier", action="store_true", default=False)
    p.add_argument("--episodes", type=int, default=0, help="0=loop forever")
    p.add_argument("--action_delay", type=int, default=1)
    p.add_argument("--save_name", type=str,default=None)
    
    p.add_argument("--env_path", type=str, default="/home/csilab/jhlee_workspace/mo_hil-serl/Real_Robo/experiments/configs/", help="e.g., 'usb'")
    p.add_argument("--env_name", type=str, default="default", help="e.g., 'usb'")
    p.add_argument(
    "--prompt", 
    type=str, 
    default="Insert USB connector into USB port", 
    help=(
        "Insert USB connector into USB port "
        "Grasp rotary switch and rotate clockwise "
        "Insert LAN cable into Ethernet port "
        "Insert plug into outlet "
        "Grasp knob and rotate clockwise "
        "Insert BNC connector and rotate clockwise "
        "Insert HDMI connector into HDMI port "
        "Grasp bar latch and rotate clockwise "
        "Insert audio jack into audio port "
        "Insert key and rotate clockwise"
    )
)
    p.add_argument("--ft_stack", type=int, default=0, help="Force-torque sensor stacking")
    p.add_argument("--action_chunk", type=int, default=5, help="action horizon length")

    p.add_argument("--residual", action="store_true", default=False)
    
    args = p.parse_args()

    global prompt
    global ft_stack
    prompt = args.prompt
    ft_stack = args.ft_stack

    if not args.residual : 
        action_chunk = 50 - args.action_chunk + 1
    else : 
        action_chunk = 5 - args.action_chunk + 1

    _path = f"{args.env_path}/{args.env_name}.json"
    print(f"Loading environment config from: {_path}")
    # get_environment 메서드 호출
    cfg = load_config(_path)
    env = build_environment(cfg)


    # client = ali.InferenceClient(server_ip=args.trainer_ip, port_num=args.trainer_port)
    client = WebsocketClientPolicy(host=args.trainer_ip, port=args.trainer_port)

    # --- action async pool ---
    action_pool = ThreadPoolExecutor(max_workers=1)

    print(f"[Env Client] connected to trainer {args.trainer_ip}:{args.trainer_port}")

    def async_predict(ob):
        return action_pool.submit(
            lambda: np.asarray(
                client.infer(npify_obs(ob, residual=args.residual))["actions"],
                dtype=np.float32,
            )
        )

    ep = 0
    trajs = []
    while True:
        prev_intervened = False
        global state_history_queue
        state_history_queue = None
        ob, info = env.reset()
        step_count = 0

        global current_episode_force_max
        current_episode_force_max = None

        # reset 직후 observation도 반영
        update_episode_force_max(ob)

        ob = npify_obs(ob, residual=args.residual)
        client.reset()

        current_chunk = pad_action_to_env(
            np.asarray(client.infer(ob)["actions"], dtype=np.float32)
        )

        action = current_chunk[0]
        ptr = 0
        future = None

        trajectory = []
        done = False
        while not done:
            # --- 1. 현재 chunk에서 action 선택 ---
            if not prev_intervened:
                action = current_chunk[ptr]

            # --- 2. 환경 step ---
            next_ob, reward, terminated, truncated, step_info = env.step(action)
            intervened = 'action_intervene' in step_info
            done = bool(terminated or truncated)
            # next_ob = npify_obs(next_ob)
            # force 저장
            update_episode_force_max(next_ob)

            # hil구현 x
            if intervened:
                action = pad_action_to_env(step_info['action_intervene']) 

            if not intervened:
                # --- 4. chunk 소진 action_delay 전에 미래 예측 시작 ---
                if prev_intervened:
                    current_chunk = pad_action_to_env(
                        np.asarray(client.infer(npify_obs(next_ob, residual=args.residual))["actions"], dtype=np.float32)
                    )
                    future = None
                    ptr = 0
                if future is None and ptr >= len(current_chunk) - action_chunk - args.action_delay:
                    future = async_predict(next_ob)

                # --- 5. chunk 완전 소진 시점: new chunk로 교체 ---
                if ptr == len(current_chunk) - action_chunk:
                    new_chunk = pad_action_to_env(future.result()) 
                    future = None

                    # 새 chunk는 new_chunk[action_delay:] 만 사용
                    delayed_tail = new_chunk[args.action_delay:]
                    current_chunk = delayed_tail
                    ptr = 0
                    continue
                else:
                    ptr += 1
            else:
                current_chunk = np.zeros_like(current_chunk)

            if args.save_name is not None:
                transition = copy.deepcopy(
                    dict(
                        observations=ob,
                        actions=action,
                        next_observations=next_ob,
                        rewards=reward,
                        masks=1.0 - done,
                        dones=done,
                        infos=step_info,
                    )
                )
                trajectory.append(transition)
            ob = next_ob
            prev_intervened = intervened
            step_count += 1
        
        if skip_key:
            episode_force_max_list.append(current_episode_force_max)
            print(f"[Episode {ep}] force abs max = {current_episode_force_max}")
            print("Saved skipped-episode force max list:", episode_force_max_list)

        if not skip_key and args.save_name is not None:
            trajs.append(trajectory)
            if len(trajs) >= 1:
                transitions = []
                for t in trajs:
                    transitions.extend(t)
                save_traj(transitions, len(trajs), args.env_name, args.save_name)
                trajs = []
        skip_key = False
        ep += 1
        if args.episodes > 0 and ep >= args.episodes:
            break

    env.close()
    action_pool.shutdown(wait=True)
    print("[Env Client] finished.")

    

if __name__ == "__main__":
    main()