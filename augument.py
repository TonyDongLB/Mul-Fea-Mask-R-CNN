import numpy as np
import pandas as pd
# import tensorflow as tf
from keras.preprocessing import image
from os.path import join
import matplotlib.pyplot as plt
import glob
import os
import cv2
from optparse import OptionParser




np.random.seed(1996)

input_size = 640

def plot_img_and_mask(img, mask):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(mask[:, :, 0])
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()


class DataAugument():
    def __init__(self, train_path, img_type):
        self.train_path = train_path
        self.train_0imgs = glob.glob(train_path + "/0/*." + img_type)  # 负样本
        self.train_1imgs = glob.glob(train_path + "/1/*." + img_type)  # 正样本

        dir = os.path.dirname(train_path)
        self.aug_train_path = os.path.join(dir, 'aug_train')
        self.img_type = img_type

        pass

    def augmentation(self, num_img=0, rotate=None, horizontal_flip=False, shift=None, zoom=None,
                     shear=None, channel_shift_range=None, gray=False, contrast=None, brightness=None,
                     saturation=None, ):

        img_paths = self.train_1imgs
        num_produced = 0
        unused_path = img_paths

        while num_produced <= num_img:

            for img_path in img_paths:
                # 已经处理过的clip
                if img_path not in unused_path:
                    continue
                # 标记是否处理过
                not_used = True
                if num_img != 0:
                    if num_produced >= num_img:
                        break

                # 读取图片部分，all_imgs为图片队列，filenames为对应的文件名（无后缀）
                all_imgs = []
                filenames = []
                filename = img_path.split('/')[-1].split('.')[0]
                num = filename.split("_")[-1]
                next_num = num

                all_imgs.append(cv2.imread(img_path))
                filenames.append(filename)
                unused_path.remove(img_path)

                # 加入其后的照片
                next_num = num
                while True:
                    next_num = str(int(next_num) + 10)
                    next_img_path = img_path.replace(num, next_num)
                    next_filename = filename.replace(num, next_num)
                    if next_img_path not in unused_path:
                        break
                    unused_path.remove(next_img_path)
                    all_imgs.append(cv2.imread(next_img_path))
                    filenames.append(next_filename)

                # 加入之前的照片
                next_num = num
                while True:
                    next_num = str(int(next_num) - 10)
                    next_img_path = img_path.replace(num, next_num)
                    next_filename = filename.replace(num, next_num)
                    if next_img_path not in unused_path:
                        break
                    unused_path.remove(next_img_path)
                    all_imgs.append(cv2.imread(next_img_path))
                    filenames.append(next_filename)


                # next_next_num = str(int(next_num) + 10)
                # next_next_img_path = img_path.replace(num, next_next_num)
                # all_imgs.append(cv2.imread(img_path))
                # filenames.append(filename)
                # if os.path.exists(next_img_path):
                #     all_imgs.append(cv2.imread(next_img_path))
                #     filenames.append(filename.replace(num, next_num))
                # if os.path.exists(next_next_img_path):
                #     all_imgs.append(cv2.imread(next_next_img_path))
                #     filenames.append(filename.replace(num, next_next_num))


                if rotate is not None:
                    if rotate is True:
                        result = self.random_rotate(all_imgs=all_imgs)
                        if result is not None:
                            writed = self.write_imgs(result, filenames, 'rotate')
                            num_produced += writed
                            not_used = False
                    else:
                        result = self.random_rotate(all_imgs=all_imgs, rotate_limit=(-rotate, rotate))
                        if result is not None:
                            writed = self.write_imgs(result, filenames, 'rotate')
                            num_produced += writed
                            not_used = False

                if horizontal_flip:
                    result = self.random_flip(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'horizontal_flip')
                        num_produced += writed
                        not_used = False

                if shift is not None:
                    if shift is True:
                        result = self.random_shift(all_imgs=all_imgs)
                        if result is not None:
                            writed = self.write_imgs(result, filenames, 'shift')
                            num_produced += writed
                            not_used = False
                    else:
                        result = self.random_shift(all_imgs=all_imgs, w_limit=(-shift, shift),
                                                   h_limit=(-shift, shift))
                        if result is not None:
                            writed = self.write_imgs(result, filenames, 'shift')
                            num_produced += writed
                            not_used = False

                if zoom:
                    result = self.random_zoom(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'zoom')
                        num_produced += writed
                        not_used = False

                if shear:
                    result = self.random_shear(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'shear')
                        num_produced += writed
                        not_used = False

                if channel_shift_range is not None:
                    result = self.random_channel_shift(all_imgs=all_imgs, limit=channel_shift_range)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'channel_shift_range')
                        num_produced += writed
                        not_used = False

                if gray:
                    result = self.random_gray(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'gray')
                        num_produced += writed
                        not_used = False

                if contrast:
                    result = self.random_contrast(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'contrast')
                        num_produced += writed
                        not_used = False

                if brightness:
                    result = self.random_brightness(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'brightness')
                        num_produced += writed
                        not_used = False

                if saturation:
                    result = self.random_saturation(all_imgs=all_imgs)
                    if result is not None:
                        writed = self.write_imgs(result, filenames, 'saturation')
                        num_produced += writed
                        not_used = False

                if not_used:
                    while True:
                        result = self.random_rotate(all_imgs=all_imgs)
                        if result is not None:
                            writed = self.write_imgs(result, filenames, 'rotate')
                            num_produced += writed
                            break

        print('Total produced {} img.'.format(num_produced))

        pass

    def write_imgs(self, all_imgs, filenames, from_fuc):
        writed = 0
        for img, filename in zip(all_imgs, filenames):
            img_path = self.aug_train_path + '/' + str(from_fuc) + '_' + filename + '.' + self.img_type
            if os.path.exists(img_path):
                return 0
            if cv2.imwrite(img_path, img):
                print('===== ' + img_path)
                writed += 1
        return writed

    def rotate(self, x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_rotate(self, all_imgs, rotate_limit=(-20, 20), u=0.5):
        if np.random.random() < u:
            theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
            result = []
            for img in all_imgs:
                result.append(self.rotate(img, theta))
            return result
        else:
            return None

    def random_flip(self, all_imgs, u=0.5):
        if np.random.random() < u:
            result = []
            for img in all_imgs:
                result.append(image.flip_axis(img, 1))
            return result
        else:
            return None

    def shift(self, x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
        h, w = x.shape[row_axis], x.shape[col_axis]
        tx = hshift * h
        ty = wshift * w
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        transform_matrix = translation_matrix  # no need to do offset
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_shift(self, all_imgs, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
        if np.random.random() < u:
            wshift = np.random.uniform(w_limit[0], w_limit[1])
            hshift = np.random.uniform(h_limit[0], h_limit[1])
            result = []
            for img in all_imgs:
                result.append(self.shift(img, wshift, hshift))
            return result
        else:
            return None

    def zoom(self, x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_zoom(self, all_imgs, zoom_range=(0.8, 1), u=0.5):
        if np.random.random() < u:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            result = []
            for img in all_imgs:
                result.append(self.zoom(img, zx, zy))
            return result
        else:
            return None

    def shear(self, x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_shear(self, all_imgs, intensity_range=(-0.5, 0.5), u=0.5):
        if np.random.random() < u:
            sh = np.random.uniform(-intensity_range[0], intensity_range[1])
            result = []
            for img in all_imgs:
                result.append(self.shear(img, sh))
            return result
        else:
            return None

    def random_channel_shift(self, all_imgs, limit, channel_axis=2, u=0.5):
        if np.random.random() < u:
            result = []
            for img in all_imgs:
                new_img = np.rollaxis(img, channel_axis, 0)
                min_x, max_x = np.min(new_img), np.max(new_img)
                channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in new_img]
                new_img = np.stack(channel_images, axis=0)
                new_img = np.rollaxis(new_img, 0, channel_axis + 1)
                result.append(new_img)
            return result
        else:
            return None

    def random_gray(self, all_imgs, u=0.5):
        if np.random.random() < u:
            coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
            result = []
            for img in all_imgs:
                gray = np.sum(img * coef, axis=2)
                new_img = np.dstack((gray, gray, gray))
                result.append(new_img)
            return result
        else:
            return None

    def random_contrast(self, all_imgs, limit=(-0.3, 0.3), u=0.5):
        if np.random.random() < u:
            alpha = 1.0 + np.random.uniform(limit[0], limit[1])
            coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
            result = []
            for img in all_imgs:
                gray = img * coef
                gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
                new_img = alpha * img + gray
                new_img = np.clip(new_img, 0., 255.)
                result.append(new_img)
            return result
        else:
            return None

    def random_brightness(self, all_imgs, limit=(-0.3, 0.3), u=0.5):
        if np.random.random() < u:
            alpha = 1.0 + np.random.uniform(limit[0], limit[1])
            result = []
            for img in all_imgs:
                new_img = alpha * img
                new_img = np.clip(new_img, 0., 255, )
                result.append(new_img)
            return result
        else:
            return None

    def random_saturation(self, all_imgs, limit=(-0.3, 0.3), u=0.5):
        if np.random.random() < u:
            alpha = 1.0 + np.random.uniform(limit[0], limit[1])
            coef = np.array([[[0.114, 0.587, 0.299]]])
            result = []
            for img in all_imgs:
                gray = img * coef
                gray = np.sum(gray, axis=2, keepdims=True)
                new_img = alpha * img + (1. - alpha) * gray
                new_img = np.clip(new_img, 0., 255.)
                result.append(new_img)
            return result
        else:
            return None

def get_args():
    parser = OptionParser()
    parser.add_option('--train_path', dest='train_path', default='/Users/apple/Documents/NN_Models/HandNet/data/train')
    parser.add_option('--img_type', dest='img_type', default='jpg')

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    args = get_args()
    aug = DataAugument(train_path=args.train_path, img_type=args.img_type, )
    aug.augmentation(num_img=0, rotate=True, horizontal_flip=True, shift=True, zoom=True,
                     shear=True, channel_shift_range=False, gray=False, contrast=False, brightness=False,
                     saturation=False
                     )
    print('!!!ALL DONE!!!')