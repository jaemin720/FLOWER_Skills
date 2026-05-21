export CUDA_VISIBLE_DEVICES=2
python server_flower copy.py \
  --checkpoint_path "/home/jack/flower_vla_calvin/logs/runs/2026-04-30/20-35-51_seed242/seed_242/saved_models/epoch=05_val_act" \
  --dataset_root /home/jack/flower_vla_calvin/custom_dataset \
  --host 0.0.0.0 \
  --port 45587
