from .base_dataset import BaseDataset
from abc import ABC
import numpy as np
import cv2 as cv
import os


class LowLevelDataset(BaseDataset, ABC):
    """
        The path of the data should be the follow form:

        self.root
            |_self.data_name
                |_Train
                    |_Blur
                        |_input_img_0.png
                        |_...
                    |_GT
                    |_Kernel
                |_Eval
                    |_Blur
                        |_input_img_0.png
                        |_...
                    |_...
    """

    def __init__(self, opt, data_name, max_num: int = 1e6):
        super(LowLevelDataset, self).__init__(opt)
        self.data_name = data_name
        self.max_num = max_num

        # data
        self.mode = "Train" if self.isTrain else "Eval"
        self.img_root = os.path.join(self.root, self.data_name, self.mode, "Blur")
        self.gt_root = os.path.join(self.root, self.data_name, self.mode, "GT")
        self.kernel_root = os.path.join(self.root, self.data_name, self.mode, "Kernel")

    def _get_pos(self, img, center_crop=False):
        h, w, _ = img.shape
        if h == self.patch_size:
            h_ = 0
        else:
            if center_crop:
                h_ = (h - self.patch_size) // 2
            else:
                h_ = np.random.randint(h - self.patch_size + 1)
        if w == self.patch_size:
            w_ = 0
        else:
            if center_crop:
                w_ = (w - self.patch_size) // 2
            else:
                w_ = np.random.randint(w - self.patch_size + 1)
        return h_, w_

    def _shift_pos(self, img, pos, pos_shift):
        h, w, *_ = img.shape
        h_shift, w_shift = pos_shift
        h_ = min(h - self.patch_size, max(0, pos[0] + h_shift))
        w_ = min(w - self.patch_size, max(0, pos[1] + w_shift))
        return h_, w_

    def _crop_img(self, img, pos=None, center_crop=False):
        h, w = pos or self._get_pos(img, center_crop)
        return img[h:h + self.patch_size, w:w + self.patch_size]

    def _crop(self, imgs, same_crop=False, center_crop=False):
        pos = self._get_pos(imgs[0], center_crop) if same_crop else None
        return list(self._crop_img(img, pos) for img in imgs)

    @staticmethod
    def _centralize(imgs):
        return list(img * 2 - 1 for img in imgs)

    @staticmethod
    def _transpose(imgs):
        return list(img.transpose(2, 0, 1) for img in imgs)

    @staticmethod
    def _transpose_back(imgs):
        return list(img.transpose(1, 2, 0) for img in imgs)

    @staticmethod
    def _resize_img(img, _h, _w):
        return cv.resize(img, (_w, _h), interpolation=cv.INTER_CUBIC)

    def _random_resize(self, imgs, low=224, high=448):

        h, w, *_ = imgs[0].shape
        if low == high:
            new = low
        else:
            new = np.random.randint(low, high) // 2 * 2
        new_w = round(new / h * w)
        new_h = round(new / w * h)
        if w >= h:
            return list(self._resize_img(img, new, new_w) for img in imgs)
        else:
            return list(self._resize_img(img, new_h, new) for img in imgs)

    @staticmethod
    def _to_float(imgs, max_value=255.):
        return list(img.astype(np.float32) / max_value for img in imgs)

    def _preprocess(self, imgs, max_value=255.):
        imgs = self._random_resize(imgs, low=256, high=360)
        imgs = self._crop(imgs, same_crop=True, center_crop=True)
        imgs = self._to_float(imgs, max_value)
        return imgs

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __len__(self):
        return min(len(os.listdir(self.img_root)), self.max_num)

    def __getitem__(self, idx):
        # img_name = os.listdir(self.img_root)[idx]
        # img = cv.cvtColor(cv.imread(os.path.join(self.img_root, img_name)), cv.COLOR_BGR2RGB) / 255.
        # gt = cv.cvtColor(cv.imread(os.path.join(self.gt_root, img_name)), cv.COLOR_BGR2RGB) / 255.
        # kernel = np.load(os.path.join(self.kernel_root, img_name[:-4] + ".npy"))[0]
        # img, gt, kernel = self._to_float([img, gt, kernel], 1.)
        # img, gt = self._transpose([img, gt])
        # return {"Data": img, "GT": gt, "Kernel": kernel, "name": img_name}


        img_name = os.listdir(self.img_root)[idx]
        img = cv.cvtColor(cv.imread(os.path.join(self.img_root, img_name)), cv.COLOR_BGR2RGB) / 255.
        gt = cv.cvtColor(cv.imread(os.path.join(self.gt_root, img_name)), cv.COLOR_BGR2RGB) / 255.
        # kernel = np.load(os.path.join(self.kernel_root, img_name[:-4] + ".npy"))[0]
        img, gt = self._to_float([img, gt], 1.)
        img, gt = self._transpose([img, gt])
        return {"Data": img, "GT": gt, "name": img_name}