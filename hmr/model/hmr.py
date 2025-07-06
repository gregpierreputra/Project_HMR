import torch
import pytorch_lightning as pl

from backbone.vitpose  import ViTBackbone
from smpl_head.transformer import SMPLTransformerDecoderHead
from discriminator import Discriminator

from losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss

from smpl.smpl_wrapper import SMPL

from typing import Tuple

# Must be defined  
_VITPOSE_BACKBONE_PRETRAINED_WEIGHTS_PATH = "./vitpose_backbone.pth"

_SMPL_MODEL_PATH = "./smpl"
_SMPL_MEAN_PARAMETERS_PATH = "./smpl_mean_params.npz"
_SMPL_JOINT_REGRESSOR_EXTRA_PATH = "./SMPL_to_J19.pkl"

_TRAIN_LEARNING_RATE = 1e-5
_TRAIN_WEIGHT_DECAY = 1e-4

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
    
    def forward_step():
        pass

    def compute_loss():
        pass
    