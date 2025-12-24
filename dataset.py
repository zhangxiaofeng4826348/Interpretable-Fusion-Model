from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import re
import SimpleITK as sitk
import pandas as pd
import random

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.label_file = label_file
        self.transform = transform

        # 读取标签文件
        self.labels = {}
        label_data = pd.read_excel(label_file)
        for index, row in label_data.iterrows():
            key = str(row['Name']).strip()  # 转成字符串，并去除可能的空格
            self.labels[key] = row['Label']

        # 获取图像文件列表并随机打乱
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        mask_file = image_file.replace(".png", ".nii")
        mask_path = os.path.join(self.mask_folder, mask_file)

        image = Image.open(image_path).convert("RGB")
        mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask = sitk.GetArrayFromImage(mask_img)
        # mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image, mask = self.transform(image, mask)


        # 读取图像对应的标签
        key = image_file.split(".")[0]
        key = key.strip()  # 去除空格
        label = torch.tensor(self.labels[key], dtype=torch.long)
        return {'image': image, 'label': label, 'mask': mask, 'filename': image_file}

