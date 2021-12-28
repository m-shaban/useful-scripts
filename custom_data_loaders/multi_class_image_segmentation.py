import os
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Segmentation_Dataset(Dataset):
    def __init__(self, dataset_dir, img_format='png', target_size=None, is_training=False):
        self.dataset_dir = dataset_dir
        self.img_list = os.listdir(dataset_dir)
        self.img_format = img_format
        if target_size is None:
            self.target_size = target_size
        else:
            self.target_size = (target_size, target_size)
        self.is_training = is_training
        self.img_list = [img_name for img_name in self.img_list if img_name.endswith(self.img_format)]


    @staticmethod
    def preprocess_train(img, mask, target_size):

        i, j, h, w = transforms.RandomCrop.get_params(img, target_size)
        img = img.crop((i, j, i + h, j + w))
        mask = mask.crop((i, j, i + h, j + w))

        img_preprocess = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.3, 0.1),
            transforms.ToTensor()
        ])
        img = img_preprocess(img)
        mask = np.array(mask)
        zero_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
        mask = np.concatenate((zero_mask, mask[:, :, 0:3]), axis=2)
        mask = np.argmax(mask, axis=2)
        return img[0:3,:,:], torch.tensor(mask)

    @staticmethod
    def preprocess_valid(img, mask, target_size):
        preprocess = transforms.Compose([
            transforms.CenterCrop(target_size),
            transforms.ToTensor()
        ])
        img = preprocess(img)
        mask = preprocess(mask)

        mask = mask.numpy()
        zero_mask = np.zeros((1, mask.shape[1], mask.shape[2]))
        mask = np.concatenate((zero_mask, mask[0:3, :, :]), axis=0)
        mask = np.argmax(mask, axis=0)
        return img[0:3,:,:], torch.tensor(mask)

    @staticmethod
    def get_data_loader(dataset, batch_size=1, drop_last=False):
        return DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=drop_last)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.img_list[idx])
        mask_path = os.path.join(self.dataset_dir, 'masks', self.img_list[idx])

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.target_size is None:
            self.target_size = (img.size[0], img.size[1])

        if self.is_training:
            rot_angle = [0, 90, 180, 270]
            random_rot = rot_angle[random.randint(0, len(rot_angle) - 1)]
            img = img.rotate(random_rot)
            mask = mask.rotate(random_rot)
            img, mask = self.preprocess_train(img, mask, self.target_size)
        else:
            img, mask = self.preprocess_valid(img, mask, self.target_size)

        return img, mask, self.img_list[idx]

if __name__ == '__main__':
    train_dir = 'train'
    valid_dir = 'valid'

    train_dataset = Segmentation_Dataset(dataset_dir=train_dir, img_format='png', target_size=512, is_training=True)
    valid_dataset = Segmentation_Dataset(dataset_dir=valid_dir, img_format='png', target_size=None, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=4)
    for img_batch, mask_batch, img_name_batch in train_loader:
        pass
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    for img_batch, mask_batch, img_name_batch in valid_loader:
        pass
    pass