python preprocess/convert_raw_pkl_to_flower.py \
  --input /data2/jack/multi_usb_p2 \
  --pkl_glob "**/*.pkl" \
  --output_root /home/jack/flower_vla_calvin/custom_ft_dataset_raw_action_alltrain \
  --task_name usb_multi \
  --include_right_ft \
  --default_instruction "Insert USB connector into USB port" \
  --action_source raw \
  --val_ratio 0
