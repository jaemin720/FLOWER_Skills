# Add --include_right_force if you also want current right force/torque appended to robot_obs.
python preprocess/convert_with_forcehistory.py \
  --input /data2/jack/2026_05_AGI/default_lsm/lsm/multi \
  --pkl_glob "**/*.pkl" \
  --output_root /home/jack/flower_vla_calvin/multi_task_dataset_raw_force_history \
  --robot_obs_format real_robot \
  --include_right_force_history \
  --action_source raw \
  --trailing_keep 5 \
  --val_ratio 0
