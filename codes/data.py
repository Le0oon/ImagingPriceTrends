import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import cv2 as cv


def parse_info(file_name):
    secu_code = file_name.split('.')[0]
    trading_day = file_name.split('-')[1].rstrip('.png')
    trading_day = trading_day[0:4] + '-' + trading_day[4:6] + '-' + trading_day[6:]

    return secu_code, trading_day


class ImageData(Dataset):
    def __init__(self, file_folder):

        positive_file = os.listdir(os.path.join(file_folder, '1'))
        negative_file = os.listdir(os.path.join(file_folder, '0'))
        self.fname = positive_file + negative_file
        self.file_dir = [os.path.join(file_folder, '1', f) for f in positive_file] + \
            [os.path.join(file_folder, '0', f) for f in negative_file]
        # self.secu_date = [parse_info(f) for f in positive_file] + \
        #     [parse_info(f) for f in negative_file]
        self.df = pd.read_hdf(os.path.join(file_folder, '..', 'price.h5'))
        attr_col = self.df.columns.tolist()[2:]
        self.label = [1 for _ in positive_file] + [0 for _ in negative_file]
        self.attr = [torch.from_numpy(np.array(self.df.loc[file, attr_col], dtype=np.float32))
                     for file in self.fname]

    def __getitem__(self, index):
        """
        return: image, label, attr
        """
        image = cv.imdecode(np.fromfile(self.file_dir[index], dtype=np.uint8), 1)
        image[image > 0] = 255
        image = torch.from_numpy(image).permute(2, 0, 1)/255

        return image, self.label[index], self.attr[index]

    def __len__(self):

        return len(self.fname)

    def __add__(self, other):

        self.file_dir += other.file_dir
        self.label += other.label
        self.attr += other.attr

        return self
