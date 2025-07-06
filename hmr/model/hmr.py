import torch
import pytorch_lightning as pl

from ..utils.geometry import perspective_projection
from backbone.vitpose  import ViTBackbone
from smpl_head.transformer import SMPLTransformerDecoderHead
from discriminator import Discriminator

from losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss

from smpl.smpl_wrapper import SMPL

from typing import Tuple, Dict

# Must be defined  
_VITPOSE_BACKBONE_PRETRAINED_WEIGHTS_PATH = "./vitpose_backbone.pth"

# Can be accessed by using wget on the following link
# e.g: wget https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2_data.tar.gz
_SMPL_MODEL_PATH = "./smpl"
_SMPL_MEAN_PARAMETERS_PATH = "./smpl_mean_params.npz"
_SMPL_JOINT_REGRESSOR_EXTRA_PATH = "./SMPL_to_J19.pkl"

_TRAIN_LEARNING_RATE = 1e-5
_TRAIN_WEIGHT_DECAY = 1e-4

_MODEL_IMAGE_SIZE = 256

_FORWARD_FOCAL_LENGTH = 5000

class HMR(pl.LightningModule):
    def __init__(self,
                 init_renderer: bool = True):
        """
        Setup the HMR module
        """
        super().__init__()

        # Toggle hyperparameter saving
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        # Create the ViTBackbone feature extractor
        # Matching the variable values to those utilized in HMR2
        self.backbone = ViTBackbone(
            img_size=(256, 192),
            embed_dim=1280,
            depth=32,
            num_heads=16,
            qkv_bias=True,
            drop_path_rate=0.55
        )
        # Load the pretrained weights for the ViTBackbone
        self.backbone.load_state_dict(torch.load(_VITPOSE_BACKBONE_PRETRAINED_WEIGHTS_PATH, 
                                                 map_location='cpu')['state_dict'])

        # Create SMPL head
        self.smpl_head = SMPLTransformerDecoderHead(smpl_mean_params_path=_SMPL_MEAN_PARAMETERS_PATH)

        # Create discriminator
        self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type = 'l1')
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type = 'l1')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        self.smpl = SMPL(
            model_path=_SMPL_MODEL_PATH,
            mean_params=_SMPL_MEAN_PARAMETERS_PATH,
            joint_regressor_extra=_SMPL_JOINT_REGRESSOR_EXTRA_PATH)
        
        # Setup renderer
        
        # Disabling automatic optimization due to use of adversarial training
        self.automatic_optimization = False


    def get_parameters(self):
        """
        Helper function to get all the parameters for the SMPLTransformerDecoderHead and the ViTBackbone for the optimizer
        """
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.backbone.parameters())

        return all_params


    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup the Optimizers for the model and discriminator

        Returns:
            Tuple of torch.optim.Optimizer for the model and discriminator respectively
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': _TRAIN_LEARNING_RATE}]

        optimizer_model = torch.optim.AdamW(params=param_groups,
                                            weight_decay=_TRAIN_WEIGHT_DECAY)
        
        optimizer_discriminator = torch.optim.AdamW(params=self.discriminator.parameters(),
                                                    lr=_TRAIN_LEARNING_RATE,
                                                    weight_decay=_TRAIN_WEIGHT_DECAY)
        
        return optimizer_model, optimizer_discriminator
    
    def forward_step(self,
                     batch: Dict) -> Dict:
        """
        Run a forward step of the network

        Arguments:
            batch (Dict)    : Dictionary containing batch data

        Returns:
            Dict: Dictionary containing the regression output
        """
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        conditioning_features = self.backbone(x[:, :, :, 32:-32])
        
        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_features)

        # Store useful regression outputs to the output dictionary
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Determine focal length based on predicted SMPL parameters 
        focal_length = _FORWARD_FOCAL_LENGTH * torch.ones(batch_size,
                                                          2,
                                                          device=device,
                                                          dtype=dtype)


        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (_MODEL_IMAGE_SIZE * pred_cam[:, 0] +1e-9)], dim=-1)

        # Save the pred_cam_t and focal_length to the output dictionary
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute the model vertices, joints, and the projected joints
        # Retrieve global orientation, body pose, betas and reshape
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)

        # Retrieve the SMPLX model outputs and predictions
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)

        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)

        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation = pred_cam_t,
                                                   focal_length = focal_length / _MODEL_IMAGE_SIZE)
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        return output


    def compute_loss():
        pass
    