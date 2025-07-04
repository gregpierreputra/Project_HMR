import numpy as np
from torch.utils.data import Dataset

class MotionCaptureDataset(Dataset):

    def __init__(self,
                 dataset_file_path: str):
        
        data = np.load(dataset_file_path)

        # Key values for the motion capture dataset
        self.pose = data['body_pose'].astype(np.float32)[:, 3:]
        self.betas = data['betas'].astype(np.float32)
        self.length = len(self.pose)

    def __getitem__(self, index: int) -> dict:
        pose = self.pose[index].copy()
        betas = self.betas[index].copy()
        
        item = {'body_pose': pose,
                'betas': betas}
        
        return item
    
    def __len__(self) -> int:
        return self.length