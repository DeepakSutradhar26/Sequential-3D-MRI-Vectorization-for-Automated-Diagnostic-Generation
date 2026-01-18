from torch.utils.data import Dataset
import nibabel as nib
import torch
import os

class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patients = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        patient_path = os.path.join(self.root_dir, self.patients[index])
        nii_files = sorted(os.listdir(patient_path))

        volumes = []
        for nii in nii_files:
            img = nib.load(nii)
            data = img.get_fdata()
            volumes.append(data)

        x = torch.stack(volumes) #(4, D, H, W)
        x = x.unsqueeze(1) # (4, 1, D, H, W)

        label = 1 if "HGG" in self.patients[index] else label = 0
        y = torch.tensor(label).unsqueeze(0)

        return x, y

