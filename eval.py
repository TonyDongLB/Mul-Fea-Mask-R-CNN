import cv2
import glob
import numpy as np
import skimage.io


def pixel_acu(pred_mask, gt_mask):
    iou = np.bitwise_and(pred_mask, gt_mask)
    pa_hand = np.sum(iou) / np.sum(pred_mask)
    pa_bg = 1 - (np.sum(pred_mask) - np.sum(iou)) / (pred_mask.size - np.sum(gt_mask))
    return pa_hand, pa_bg


def iou(pred_mask, gt_mask):
    iou = np.bitwise_and(pred_mask, gt_mask)
    all = np.bitwise_or(pred_mask, gt_mask)
    return np.sum(iou) / np.sum(all)


if __name__ == '__main__':
    pa_hand_scores = []
    pa_bg_scores = []
    iou_scores = []
    pred_paths = glob.glob('result/*.jpg')
    for pred_path in pred_paths:
        pred_mask = skimage.io.imread(pred_path)
        file_name = pred_path.split('/')[-1]
        gt_mask = skimage.io.imread('hand_instance/mask_labels/val/' + file_name)
        if gt_mask is None:
            print(file_name + 'is None')
        pred_mask = np.clip(pred_mask, 0, 1)
        gt_mask = np.clip(gt_mask, 0, 1)
        pa_hand, pa_bg = pixel_acu(pred_mask, gt_mask)
        pa_hand_scores.append(pa_hand)
        pa_bg_scores.append(pa_bg)
        iou_scores.append(iou(pred_mask, gt_mask))

    print('Hand Pixel Accuracy is ' + str(sum(pa_hand_scores) / len(pa_hand_scores)))
    print('BackGround Pixel Accuracy is ' + str(sum(pa_bg_scores) / len(pa_bg_scores)))
    print('IOU is ' + str(sum(iou_scores) / len(iou_scores)))
