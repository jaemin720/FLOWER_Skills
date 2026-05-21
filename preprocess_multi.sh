python /home/jack/flower_vla_calvin/preprocess/convert_multitask_raw_pkl_to_flower_act_norm.py \
  --input /data2/jack/2026_04_AGI/2026_04_agi_multi \
  --pkl_glob "**/*.pkl" \
  --output_root /home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain_act_norm \
  --include_right_ft \
  --action_source raw \
  --val_ratio 0
