import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import torchvision.transforms as transforms

class MAHCD(Dataset):
    # img1-sat img2-uav
    def __init__(self, dir_chin_data, txt_dir="str", mode=None):

        self.dir = dir_chin_data
        self.mode = mode
        self.txt_path = txt_dir

        if self.mode is not None:
            txt_trainval = os.path.join(self.txt_path, mode + ".txt")
            with open(txt_trainval, 'r') as file:
                self.images = [line.strip() for line in file]
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'train' or 'val'.")

        # Replace it with this using the MAHCD dataset
        self.sat_mean = np.array([107, 117, 115], np.uint8)
        self.uav_mean = np.array([109, 130, 128], np.uint8)


        #Replace it with this using the HTCD dataset
        #self.sat_mean = np.array([66, 71, 74], np.uint8)
        #self.uav_mean = np.array([73, 81, 79], np.uint8)


    def __getitem__(self, idx):
        # img1-sat img2-uav

        filename = self.images[idx]

        img1_file = os.path.join(self.dir, 'sat', filename)
        img2_file = os.path.join(self.dir, 'uav', filename)
        lbl_file = os.path.join(self.dir, 'label', filename)

        img1 = cv2.imread(img1_file)
        if img1 is None:
            print("###", img1_file)
        img1 -= self.sat_mean
        #img1 = cv2.resize(img1, (256, 256))
        img_size = img1.shape[:2]

        img2 = cv2.imread(img2_file)
        if img2 is None:
            print("###", img2_file)

        img2 = cv2.resize(img2, (1024, 1024))
        img2 -= self.uav_mean

        lbl = cv2.imread(lbl_file, cv2.IMREAD_UNCHANGED)

        if lbl is None:
            print("###", lbl_file)
        lbl = cv2.resize(lbl, img_size, interpolation=cv2.INTER_NEAREST)

        self.sample = [img1,img2,lbl]
        if self.mode == "train":
            img1, img2, lbl = train_transforms(self.sample)
        else:
            img1, img2, lbl = val_transforms(self.sample)
        return img1, img2, lbl, filename

    def __len__(self):
        return len(self.images)


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample[0]
        img2 = sample[1]
        mask = sample[2]
        #print("shuipin",random.random())
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 1)  # 1 表示水平翻转
            img2 = cv2.flip(img2, 1)
            mask = cv2.flip(mask, 1)

        return [img1, img2, mask]


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample[0]
        img2 = sample[1]
        mask = sample[2]
        #print("chuizhi", random.random())
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 0)  # 1 表示水平翻转
            img2 = cv2.flip(img2, 0)
            mask = cv2.flip(mask, 0)

        return [img1, img2, mask]

class ToTensor(object):
    def __call__(self, sample):
        img1 = sample[0]
        img2 = sample[1]
        mask = sample[2]
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1)) / 128.0
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1)) / 128.0
        mask = np.asarray(mask)
        #print("to tesnor")

        return [img1, img2, mask]


train_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()])
val_transforms = transforms.Compose([ToTensor()])

