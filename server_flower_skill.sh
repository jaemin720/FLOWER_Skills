export CUDA_VISIBLE_DEVICES=3
python server_flower_skill.py \
  --checkpoint_path "/data2/jack/log/runs/2026-05-27/00-16-04_seed242/seed_242/saved_models/epoch=11_train/action_loss=0.2474.ckpt" \
  --dataset_root /home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain \
  --host 0.0.0.0 \
  --port 45587 \
  --action_horizon 32
