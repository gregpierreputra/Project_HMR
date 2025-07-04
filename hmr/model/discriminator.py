import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        """
        Pose and Shape Discriminator
        Determines if the presented skinned multi-person linear (SMPL) model's poses, betas, and joints correspond to a real human shape and pose
        
        Inferred parameters sent to the discriminator to determine if the 3D parameters are valid human shapes and poses from the unpaired data.
        Acts as weak supervision for natural images without ground truth 3D annotations.

        Discriminator for pose and shape are independent.
        
        Pose discriminators trained for each joint rotation, and learning the angle limits for each joint.
        Share common feature space of rotation matrices

        Input for each discriminator:
        - 10 dimensional : Betas
        - 9 dimensional  : Each joint
        - 9 dimensional  : All joints

        Output for each discriminator:
        - [0, 1] : Betas
        - [0, 1] : Each joint
        - [0, 1] : All joints

        Represents the probability that the final estimate came from the data

        -- Architecture --
        Per-part relative rotation:
        - 2 layers with 32 hidden neurons
            - Output then sent to 23 discriminators (correspond to the 23 joints of a SMPL model) that output 1-D values
        - ReLU activations except the final layer
        - AdamW optimizer
        - Xavier uniform for weights initialization        
        
        Shape parameters (betas):
        - 2 layers with 10, 5, and 1 neurons
        - ReLU activations except the final layer
        - AdamW optimizer
        - Xavier uniform for weights initialization

        Overall pose distribution:
        - 2 layers of 1024 neurons each
            - Outputs a 1D value
        - ReLU activations except the final layer
        - AdamW optimizer
        - Xavier uniform for weights initialization

        Paper URL: https://arxiv.org/pdf/1712.06584
        - Greg / 4 July 2025
        """
        super(Discriminator, self).__init__()

        # Number of joints to discriminate against - corresponds to final layer
        self.number_of_joints = 23

        # ReLU activation layer
        self.relu = nn.ReLU(inplace=True)

        # -- Per-Part Relative Rotation --
        # a 3D rotation for each of the 23 joints of a SMPL model
        pose_out = []

        # 1st Layer
        self.discriminator_pose_conv1 = nn.Conv2d(in_channels = 9, 
                                                  out_channels = 32, 
                                                  kernel_size = 1)
        nn.init.xavier_uniform_(self.discriminator_pose_conv1.weight)
        nn.init.zeros_(self.discriminator_pose_conv1.bias)

        # 2nd Layer
        self.discriminator_pose_conv2 = nn.Conv2d(in_channels = 32,
                                                  out_channels = 32,
                                                  kernel_size = 1)
        nn.init.xavier_uniform_(self.discriminator_pose_conv2.weight)
        nn.init.zeros_(self.discriminator_pose_conv2.bias)

        # Output Layer
        # Linear layer stored in dictionaries that output [0, 1]
        for _ in range(self.number_of_joints):
            pose_out_temperature = nn.Linear(in_features = 32,
                                             out_features = 1)
            nn.init.xavier_uniform_(pose_out_temperature.weight)
            nn.init.zeros_(pose_out_temperature.bias)

            pose_out.append(pose_out_temperature)
        
        self.pose_out = nn.ModuleList(pose_out)


        # -- Shape parameters (betas) --
        # 1st Layer
        self.discriminator_betas_fc1 = nn.Linear(in_features = 10, 
                                   out_features = 10)
        nn.init.xavier_uniform_(self.discriminator_betas_fc1.weight)
        nn.init.zeros_(self.discriminator_betas_fc1.bias)

        # 2nd Layer
        self.discriminator_betas_fc2 = nn.Linear(in_features = 10, 
                                   out_features = 5)
        nn.init.xavier_uniform_(self.discriminator_betas_fc2.weight)
        nn.init.zeros_(self.discriminator_betas_fc2.bias)

        # Output Layer
        self.discriminator_betas_out = nn.Linear(in_features = 5, 
                                   out_features = 1)
        nn.init.xavier_uniform_(self.discriminator_betas_out.weight)
        nn.init.zeros_(self.discriminator_betas_out.bias)


        # -- Overall body pose distribution --
        # 1st Layer
        self.discriminator_body_fc1 = nn.Linear(in_feautres = 32 * self.number_of_joints, 
                                                out_features = 1024)
        nn.init.xavier_uniform_(self.discriminator_body_fc1.weight)
        nn.init.zeros_(self.discriminator_body_fc1.bias)

        # 2nd Layer
        self.discriminator_body_fc2 = nn.Linear(in_features = 1024,
                                                out_features = 1024)
        nn.init.xavier_uniform_(self.discriminator_body_fc2.weight)
        nn.init.zeros_(self.discriminator_body_fc2.bias)

        # Output Layer
        self.discriminator_body_out = nn.Linear(in_features = 1024,
                                                out_features = 1)
        nn.init.xavier_uniform_(self.discriminator_body_out.weight)
        nn.init.zeros_(self.discriminator_body_out.bias)


    def forward(self,
                poses: torch.Tensor,
                betas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Refer to the SMPL Head module on the reasoning behind the tensor shapes and joints.

        Arguments:
            poses (torch.Tensor): Tensor of shape (B, 23, 3, 3) containing a batch of SMPL body poses (excluding the global orientation).
            betas (torch.Tensor): Tensor of shape (B, 10) containign a batch of SMPL beta coefficients.
        
        Returns:
            torch.Tensor: Discriminator output with shape (B, 25)
        """
        # Reshaping
        
        # Poses: B x number_of_joints x 1 x 9
        poses = poses.reshape(-1, self.number_of_joints, 1, 9)
        bin = poses.shape[0]

        # Poses: B x 9 x number_of_joints x 1
        poses = poses.permute(0, 3, 1, 2).contiguous()

        # -- Per-Part Relative Rotation --
        poses = self.discriminator_pose_conv1(poses)
        poses = self.relu(poses)
        poses = self.discriminator_pose_conv2(poses)
        poses = self.relu(poses)

        poses_out = []
        for i in range(self.number_of_joints):
            poses_out_ = self.pose_out[i](poses[:, :, i, 0])
            poses_out.append(poses_out_)
        
        poses_out = torch.cat(poses_out,
                              dim = 1)

        # -- Shape parameters (betas) --
        betas = self.discriminator_betas_fc1(betas)
        betas = self.relu(betas)
        betas = self.discriminator_betas_fc2(betas)
        betas = self.relu(betas)
        betas_out = self.discriminator_betas_out(betas)

        # -- Overall body pose distribution --
        poses = poses.reshape(bin, -1)
        poses_all = self.discriminator_body_fc1(poses)
        poses_all = self.relu(poses_all)
        poses_all = self.discriminator_betas_fc2(poses_all)
        poses_all = self.relu(poses_all)
        poses_all_out = self.discriminator_body_out(poses_all)

        discriminators_out = torch.cat((poses_out, betas_out, poses_all_out),
                                       dim = 1)
        
        return discriminators_out