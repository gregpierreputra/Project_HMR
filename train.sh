#!/bin/bash

TRAIN_DATASET="COCO-TRAIN-2014-WMASK-PRUNED"
VAL_DATASET="COCO-VAL"
MOCAP_PATH="/opt/ml/data/MDN/hmr2_training_data/cmu_mocap.npz"
AMASS_PATH="/opt/ml/data/MDN/hmr2_training_data/amass_poses_hist100_SMPL+H_G.npy"

SMPL_MODEL_PATH="/opt/ml/misc/MDN/HMR2/mpips_smplify_public_v2/smplify_public/code/models"
SMPL_MEAN_PARAMS="/opt/ml/misc/MDN/HMR2/smpl_mean_params.npz"
SMPL_JOINT_REGRESSOR="/opt/ml/misc/MDN/HMR2/SMPL_to_J19.pkl"
VITPOSE_BACKBONE="/opt/ml/misc/MDN/HMR2/vitpose_small_backbone.pth"

CHECKPOINT_PATH="./checkpoints/"

ACCELERATOR="gpu"
MAX_EPOCHS=2
BATCH_SIZE=4
NUM_WORKERS=4
PREFETCH_FACTOR=2
CHECKPOINT_TOPK=1
LR=1e-5
WD=1e-4
GRAD_CLIP=1.0
FOCAL_SCALE=5000

LOG_EVERY_N_STEPS=10


# Run the Python script with arguments
python train.py \
  --train_dataset_name "$TRAIN_DATASET" \
  --val_dataset_name "$VAL_DATASET" \
  --mocap_datafile_path "$MOCAP_PATH" \
  --amass_poses_hist100_path "$AMASS_PATH" \
  --smpl_model_path "$SMPL_MODEL_PATH" \
  --smpl_mean_params_path "$SMPL_MEAN_PARAMS" \
  --smpl_joint_regressor_extra_path "$SMPL_JOINT_REGRESSOR" \
  --vitpose_backbone_pretrained_path "$VITPOSE_BACKBONE" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --accelerator "$ACCELERATOR" \
  --max_epochs "$MAX_EPOCHS" \
  --log_every_n_steps "$LOG_EVERY_N_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --checkpoint_topk "$CHECKPOINT_TOPK" \
  --learning_rate "$LR" \
  --weight_decay "$WD" \
  --grad_clip_val "$GRAD_CLIP" \
  --focal_length_scale "$FOCAL_SCALE"
