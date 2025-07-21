import os
from pathlib import Path

from hmr.model.hmr import HMRLightningModule
from hmr.config import get_config

def check_smpl_exists(smpl_file_location: str):
    smpl_model_exists = os.path.exists(smpl_file_location)

    if not any(smpl_model_exists):
        raise FileNotFoundError("Did not find SMPL model in location: {}. Please dowmnload it from https://smplify.is.tue.mpg.de/ and place it in the appropriate location"
                                .format(smpl_file_location))

    return True

def load_HMR(checkpoint_path: str,
             smpl_file_location: str,
             smpl_model_path: str,
             smpl_joint_regressor_extra_path: str,
             smpl_mean_params_path: str):
    
    # Retrieve the configs stored in the yaml file
    model_cfg = str(Path(checkpoint_path).parent.parent / 'demo.yaml')

    model_cfg = get_config(
        model_cfg,
        smpl_model_path,
        smpl_joint_regressor_extra_path,
        smpl_mean_params_path)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Check to ensure that an SMPL model exists
    check_smpl_exists(smpl_file_location)

    # Load the model utilizing the checkpoint path defined in the argument parser
    model = HMRLightningModule.load_from_checkpoint(
        checkpoint_path, 
        strict=False, 
        cfg=model_cfg)
    
    return model, model_cfg