# Default arguments
# model_config_path="/opt/ml/misc/MDN/HMR2/model_config.yaml"
# smpl_model_path="/opt/ml/misc/MDN/HMR2/mpips_smplify_public_v2/smplify_public/code/models"
# smpl_mean_params_path="/opt/ml/misc/MDN/HMR2/smpl_mean_params.npz"
# smpl_joint_regressor_extra_path="/opt/ml/misc/MDN/HMR2/SMPL_to_J19.pkl"
# vitpose_backbone_pretrained_path="/opt/ml/misc/MDN/HMR2/vitpose_small_backbone.pth"
# smpl_pkl_file="/opt/ml/misc/MDN/HMR2/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
# checkpoint="/opt/ml/misc/MDN/HMR2/checkpoint_file.ckpt"
# img_folder="/opt/ml/misc/MDN/HMR2/input_images"
# out_folder="/opt/ml/misc/MDN/HMR2/output_images"

# Default arguments - Greg
model_config_path="/home/greg/Monash_MDN_Projects/HMR/model_config.yaml"
smpl_model_path="/home/greg/Monash_MDN_Projects/HMR"
smpl_mean_params_path="/home/greg/Monash_MDN_Projects/HMR/smpl_mean_params.npz"
smpl_joint_regressor_extra_path="/home/greg/Monash_MDN_Projects/HMR/SMPL_to_J19.pkl"
vitpose_backbone_pretrained_path="/home/greg/Monash_MDN_Projects/HMR/vitpose_small_backbone.pth"
smpl_pkl_file="/home/greg/Monash_MDN_Projects/HMR/SMPL_NEUTRAL.pkl"
checkpoint="/home/greg/Monash_MDN_Projects/HMR/epoch_0086-step_000086130.ckpt"
img_folder="/home/greg/Monash_MDN_Projects/HMR/input_images"
out_folder="/home/greg/Monash_MDN_Projects/HMR/output_images"
side_view="False"
top_view="False"
full_frame="False"
save_mesh="True"
detector="vitdet"
batch_size=1

# Load overrides from .env file if it exists
if [ -f .env ]; then
    echo "Loading overrides from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Launch Python script with all arguments
python demo.py \
    --model_config_path "$model_config_path" \
    --smpl_model_path "$smpl_model_path" \
    --smpl_mean_params_path "$smpl_mean_params_path" \
    --smpl_joint_regressor_extra_path "$smpl_joint_regressor_extra_path" \
    --vitpose_backbone_pretrained_path "$vitpose_backbone_pretrained_path" \
    --checkpoint "$checkpoint" \
    --smpl_pkl_file "$smpl_pkl_file" \
    --img_folder "$img_folder" \
    --out_folder "$out_folder" \
    --detector "$detector"
    # --side_view "$side_view" \
    # --top_view "$top_view" \
    # --full_frame "$full_frame" \
    # --save_mesh "$save_mesh" \