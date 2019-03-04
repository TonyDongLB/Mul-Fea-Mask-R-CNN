import torch
import torch.utils.data
from pycocotools import mask as maskUtils
import os

from config import Config
import cv2
import glob
import random
import numpy as np
import json
from PIL import Image
import skimage.io
import utils
import model
import model as modellib
from model import build_rpn_targets, mold_image, compose_image_meta
from keras.preprocessing import image


# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class HandConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hand"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # hand has 80 classes

    # TODO 以下都是新改的

    # Pooled ROIs
    POOL_SIZE = 7 # 给分类器用的
    MASK_POOL_SIZE = 28 # 传给MSAK分支的feature map 大小
    MASK_SHAPE = [56, 56] # 生成的最终mask大小

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (112, 112)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 768
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported



#############################
###      自定义的数据       ###
#############################

class Hand(torch.utils.data.Dataset):
    """
    输出类似于COCO数据格式
    """
    def __init__(self, mode, config, augment=True):
        super(Hand, self).__init__()
        self.config = config
        self.augment = augment
        if mode == 'train':
            self.imgs = glob.glob('./hand_instance/images/train/*.jpg')
            self.jsons = glob.glob('./hand_instance/jsons/*.json')
            self.num_of_imgs = len(self.imgs)
            self.num_of_jsons = len(self.jsons)
        elif mode == 'val':
            self.imgs = glob.glob('./hand_instance/images/val/*.jpg')
            self.num_of_imgs = len(self.imgs)
            self.num_of_jsons = 0
        else:
            self.imgs = glob.glob('./hand_instance/images/test/*.jpg')
            self.num_of_imgs = len(self.imgs)
            self.num_of_jsons = 0
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, index):
        index = index // 6
        config = self.config
        if index < self.num_of_imgs:
            img_path = self.imgs[index]
            mask_path = 'hand_instance/mask_labels/' + img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
            instance_masks = []
            image = skimage.io.imread(img_path)
            instance_masks.append(skimage.io.imread(mask_path))
            class_ids = []
            class_ids.append(1)
            gt_class_ids = np.array(class_ids, dtype=np.int32)
        else:
            json_index = index - self.num_of_imgs
            json_path = self.jsons[json_index]
            img_info = json.load(open(json_path, 'r'))
            file_name = img_info["file"]
            img_path = 'hand_instance/all-images' + '/' + file_name
            image = skimage.io.imread(img_path)
            objects = img_info['objects']
            instance_masks = []
            class_ids = []
            for obj in objects:
                polygon = obj['polygon']
                segm = []
                for point in polygon:
                    segm.append(point['x'])
                    segm.append(point['y'])
                if len(segm) < 16:
                    continue
                height, width = image.shape[:2]
                rles = maskUtils.frPyObjects([segm], height, width)
                rle = maskUtils.merge(rles)
                m = maskUtils.decode(rle)
                instance_masks.append(m)
                class_ids.append(1)
            gt_class_ids = np.array(class_ids, dtype=np.int32)

        for i in range(len(instance_masks)):
            instance_masks[i] = instance_masks[i][:,:, np.newaxis]

        # 数据增广
        if self.augment:
            choice_1 = random.randint(0, 2)
            if choice_1 == 1:
                instance_masks.append(image)
                instance_masks = self.random_flip(instance_masks, u=1)
                instance_masks, image = instance_masks[: -1], instance_masks[-1]
            if choice_1 == 2:
                instance_masks.append(image)
                instance_masks = self.random_rotate(instance_masks, u=1)
                instance_masks, image = instance_masks[: -1], instance_masks[-1]
            choice_2 = random.randint(0, 3)
            if choice_2 == 1:
                image = self.random_brightness([image], u=1)[0]
            if choice_2 == 2:
                image = self.random_saturation([image], u=1)[0]
            if choice_2 == 3:
                image = self.random_contrast([image], u=1)[0]

        # cv2.imwrite('image.jpg', image)
        # for i in range(len(instance_masks)):
        #     cv2.imwrite(str(i)+'.jpg', instance_masks[i] * 255)
        for i in range(len(instance_masks)):
            instance_masks[i] = instance_masks[i][:,:, 0]


        gt_masks = np.stack(instance_masks, axis=2)

        shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        gt_masks = utils.resize_mask(gt_masks, scale, padding)

        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        gt_boxes = utils.extract_bboxes(gt_masks)

        # 展示数据
        # self.display_instances(image, gt_boxes, gt_masks, class_ids)

        if self.config.USE_MINI_MASK:
            gt_masks = utils.minimize_mask(gt_boxes, gt_masks, config.MINI_MASK_SHAPE)

        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        images = mold_image(image.astype(np.float32), self.config)

        image_metas = compose_image_meta(0, shape, window, [1, 1])
        # Convert
        images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

    def display_instances(self, image, boxes, masks, class_ids):
        import visualize
        class_names = ['BG', 'Hand']
        class_ids = np.array(class_ids, dtype=np.int32)
        new_boxes = []
        for box in boxes:
            new_box = list(box)
            new_box.append(1)
            new_boxes.append(new_box)
        new_boxes = np.stack(new_boxes, axis=0)
        visualize.display_instances(image, boxes, masks, class_ids, class_names)
        return


    def rotate(self, x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0.):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_rotate(self, all_imgs, rotate_limit=(-10, 10), u=0.5):
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

    def __len__(self):
        return (self.num_of_imgs + self.num_of_jsons) * 6



############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("--command",
                        default='train',
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    # TODO 数据库
    parser.add_argument('--dataset',
                        default='/home/dl/Documents/coco/images',
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')

    parser.add_argument('--model', required=False,
                        default='coco',
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")

    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HandConfig()
    else:
        class InferenceConfig(HandConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN_Hand(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN_Hand(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = Hand(mode='train',config=config)

        # Validation dataset
        dataset_val = Hand(mode='val',config=config)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        import coco
        dataset_val = coco.CocoDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        coco.evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
        coco.evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

