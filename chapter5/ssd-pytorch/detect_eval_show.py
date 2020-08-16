"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from utils.augmentations import SSDAugmentation
import torch.utils.data as data
from torchvision import transforms
from visdom import Visdom
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.patches as patches
import random

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

parser.add_argument('-l', '--label', action='store_true',
                    help='show label in output image')

parser.add_argument('-b', '--bbox', action='store_true',
                    help='show bbox in output image')

parser.add_argument('-s', '--score', action='store_true',
                    help='show bbox in score')

parser.add_argument('num', nargs='?', default=4, type=int,
                    help='num of images to show')

parser.add_argument('--thresh', default=0.15, type=float,
                    help='thresh score value')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2012'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
# dataset_mean = (104/256.0, 117/256.0, 123/256.0)
set_type = 'val'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.13):
    num_images = min(len(dataset), args.num)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in range(num_images)]
    imgs_arr = []

    print('np.array(all_boxes).shape: ', np.array(all_boxes).shape)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        rand_idx = random.randint(0, (len(dataset) - 1))
        # rand_idx = i
        _, _, h, w = dataset.pull_item(rand_idx)
        im = dataset.pull_image(rand_idx)
        print('orig im.shape: ', im.shape)
        imgs_arr.append(im)

        print('transform(im)[0].shape: ', transform(im)[0].shape)
        x = torch.from_numpy(transform(im)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        # detections = net(x).data
        detections = net(x)
        detect_time = _t['im_detect'].toc(average=False)
        # print('image num: ', i)
        # print("detections.shape: ", detections.shape)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0] > thresh
            dets = dets[mask]
            # print("dets.shape: ", dets.shape)
            # print("mask.shape: ", mask.shape)
            if len(dets) > 0:
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].detach().cpu().numpy()
                boxes = boxes.detach().cpu().numpy()
                # print('image num: ' + str(i) + ' class: ' + str(j -1) + ' name: ' + labelmap[j-1])
                classes = np.full_like(scores, fill_value=(j-1), dtype=int)
                # print('classes: \n', classes)
                # print('scores: \n', scores)
                # print('scores[:, np.newaxis]: \n', scores[:, np.newaxis])
                # print('boxes: \n', boxes)
                # print('\n')

                # [obj count] , [class_idx, score, xmin, ymin, xmax, ymax]
                class_objs = np.hstack((classes[:, np.newaxis],
                                        scores[:, np.newaxis],
                                        boxes)).astype(float, copy=False)
                # print('class_objs.shape: ', class_objs.shape)
                # print('class_objs: \n', class_objs)
                all_boxes[i].append(class_objs)
            # all_boxes[j][i] = cls_dets

        print('all_boxes[%d] len: %d '%(i, len(all_boxes[i])))

    print('np.array(all_boxes).shape: ', np.array(all_boxes).shape)
    show_detect_result(imgs_arr=imgs_arr, all_boxes=all_boxes)

color_list = ['red', 'aqua', 'darkorange','lime', 'blue', 'purple']

def show_detect_result(imgs_arr=[], all_boxes=[]):
    trans_func = transforms.ToPILImage()
    imgs_one_line = int(len(imgs_arr) / 2 + (len(imgs_arr) % 2))
    for idx, img in enumerate(imgs_arr):
        # img_plt = trans_func(img).convert('RGB')
        axes = plt.subplot(2, imgs_one_line, (idx + 1))
        i = 0
        objs = all_boxes[idx]
        print('len(objs): ', len(objs))
        for obj_class in objs:
            # print('item[0]: ', int(item[0].item()))
            for item in obj_class:
                print('item.shape: ', item.shape)
                name = labelmap[int(item[0])]
                score = item[1]
                xmin = int(item[2])
                ymin = int(item[3])
                xmax = int(item[4])
                ymax = int(item[5] )
                # print('name: ', name)
                # print('(xmin, ymin, xmax, ymax): ', xmin, ymin, xmax, ymax)
                i += 1
                i %= len(color_list)
                rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                        linewidth=2, edgecolor=color_list[i], fill=False)
                if args.score == True:
                    axes.text(rect.xy[0], rect.xy[1], str(score),
                            va='center', ha='center', color=color_list[i],
                            bbox=dict(facecolor='w'))
                if args.bbox == True:
                    axes.add_patch(rect)

                if args.label == True:
                    axes.text(rect.xy[0], rect.xy[1], name,
                            va='center', ha='center', color='k',
                            bbox=dict(facecolor='w'))
        plt.imshow(img)
        plt.axis('off')
        plt.ioff()

    viz = Visdom(env='ssd_obj_detect')
    viz.matplot(plt)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(args.voc_root, [(YEAR, set_type)],
                           None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    print('net.size: ', net.size)
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.thresh)
