export CUDA_VISIBLE_DEVICES=3
python server_flower_copy.py \
  --checkpoint_path "/data2/jack/log/runs/2026-05-25/17-11-56_seed242/seed_242/saved_models/last.ckpt" \
  --dataset_root /home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain \
  --host 0.0.0.0 \
  --port 45587 \
  --action_horizon 32
