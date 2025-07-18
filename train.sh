#!/bin/bash

# Default arguments
train_dataset_name="COCO-TRAIN-2014-WMASK-PRUNED"
val_dataset_name="COCO-VAL"
mocap_datafile_path="/opt/ml/data/MDN/hmr2_training_data/cmu_mocap.npz"
amass_poses_hist100_path="/opt/ml/data/MDN/hmr2_training_data/amass_poses_hist100_SMPL+H_G.npy"
smpl_model_path="/opt/ml/misc/MDN/HMR2/mpips_smplify_public_v2/smplify_public/code/models"
smpl_mean_params_path="/opt/ml/misc/MDN/HMR2/smpl_mean_params.npz"
smpl_joint_regressor_extra_path="/opt/ml/misc/MDN/HMR2/SMPL_to_J19.pkl"
vitpose_backbone_pretrained_path="/opt/ml/misc/MDN/HMR2/vitpose_small_backbone.pth"
checkpoint_path="./checkpoints/"
accelerator="cpu"
max_epochs=100
log_every_n_steps=50
batch_size=64
num_workers=8
prefetch_factor=2
mocap_num_train_samples=10
checkpoint_topk=2
mlflow_experiment_name="hmr"
mlflow_uri="http://localhost:5000"
mlflow_run_name=""
mlflow_log_model="all"
learning_rate=1e-4
weight_decay=1e-4
grad_clip_val=1.0
focal_length_scale=5000
loss_3d_keypoint_weight=0.05
loss_2d_keypoint_weight=0.01
loss_global_orient_weight=0.001
loss_body_pose_weight=0.001
loss_betas_weight=0.0005
loss_adversarial_weight=0.0005

# Load overrides from .env file if it exists
if [ -f .env ]; then
    echo "Loading overrides from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Launch Python script with all arguments
python train.py \
    --train_dataset_name "$train_dataset_name" \
    --val_dataset_name "$val_dataset_name" \
    --mocap_datafile_path "$mocap_datafile_path" \
    --amass_poses_hist100_path "$amass_poses_hist100_path" \
    --smpl_model_path "$smpl_model_path" \
    --smpl_mean_params_path "$smpl_mean_params_path" \
    --smpl_joint_regressor_extra_path "$smpl_joint_regressor_extra_path" \
    --vitpose_backbone_pretrained_path "$vitpose_backbone_pretrained_path" \
    --checkpoint_path "$checkpoint_path" \
    --accelerator "$accelerator" \
    --max_epochs "$max_epochs" \
    --log_every_n_steps "$log_every_n_steps" \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --prefetch_factor "$prefetch_factor" \
    --mocap_num_train_samples "$mocap_num_train_samples" \
    --checkpoint_topk "$checkpoint_topk" \
    --mlflow_experiment_name "$mlflow_experiment_name" \
    --mlflow_uri "$mlflow_uri" \
    --mlflow_run_name "$mlflow_run_name" \
    --mlflow_log_model "$mlflow_log_model" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --grad_clip_val "$grad_clip_val" \
    --focal_length_scale "$focal_length_scale" \
    --loss_3d_keypoint_weight "$loss_3d_keypoint_weight" \
    --loss_2d_keypoint_weight "$loss_2d_keypoint_weight" \
    --loss_global_orient_weight "$loss_global_orient_weight" \
    --loss_body_pose_weight "$loss_body_pose_weight" \
    --loss_betas_weight "$loss_betas_weight" \
    --loss_adversarial_weight "$loss_adversarial_weight" \
    "$@"
