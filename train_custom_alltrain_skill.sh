#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain
QUEST_SKILL_LABEL_PATH=${DATA_ROOT}/training/quest_skill_indices_s32_d4.npy
PRETRAINED_MODEL_PATH=/home/jack/flower_vla_calvin/flower_vla_pret/360000_model_weights.pt

if [[ ! -f "${QUEST_SKILL_LABEL_PATH}" ]]; then
  echo "Missing QueST skill labels: ${QUEST_SKILL_LABEL_PATH}"
  echo "Create them first with:"
  echo "  conda activate quest"
  echo "  cd ${SCRIPT_DIR}"
  echo "  python preprocess/precompute_quest_skill_labels.py \\"
  echo "    --dataset_dir ${DATA_ROOT}/training \\"
  echo "    --quest_checkpoint /home/jack/quest_practice/QueST/experiments/flower_calvin/multi_task_raw_action_alltrain/quest/flower_calvin_skill_vae/block_32_ds_4_ep200/0/stage_0/multitask_model_epoch_0200.pth"
  exit 1
fi

if [[ ! -f "${PRETRAINED_MODEL_PATH}" ]]; then
  echo "Missing FLOWER pretrained weights: ${PRETRAINED_MODEL_PATH}"
  exit 1
fi

python flower/training_calvin.py --config-name config_custom_calvin_alltrain_skill \
  root_data_dir=/home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain \
  batch_size=8 \
  num_workers=8 \
  max_epochs=20 \
  model.pretrained_model_path="${PRETRAINED_MODEL_PATH}" \
  model.use_quest_skill=true \
  model.quest_skill_label_path="${QUEST_SKILL_LABEL_PATH}" \
  model.quest_autoencoder_ckpt=null
