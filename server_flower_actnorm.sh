export CUDA_VISIBLE_DEVICES=3

python server_flower_copy_actnorm.py \
  --checkpoint_path /data2/jack/log/runs/2026-05-13/21-22-18_seed242/seed_242/saved_models/epoch=19_train/action_loss=0.1639.ckpt \
  --dataset_root /home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain_act_norm \
  --host 0.0.0.0 \
  --port 45587 \
  --action_stats_path /home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain_act_norm/action_statistics.npz