python preprocess/eda_action_normalization.py \
  --input /data2/jack/2026_04_AGI/2026_04_agi_multi \
  --pkl_glob "**/*.pkl" \
  --action_source raw \
  --stats_path /home/jack/flower_vla_calvin/custom_multitask_dataset/action_statistics.npz \
  --save_json /home/jack/flower_vla_calvin/action_eda.json
