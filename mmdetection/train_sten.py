# Check Pytorch installation
import torch, torchvision
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print("mmdetection:",mmdet.__version__)

# Check mmcv installation
import mmcv
print("mmcv:",mmcv.__version__)

# Check mmengine installation
import mmengine
print("mmengine:",mmengine.__version__)

from mmengine.runner import Runner
from mmengine import Config
from mmengine.runner import set_random_seed

import numpy as np
import random
import os




cfg = Config.fromfile('/home/sbhandari/seg/mmdetection/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco copy.py')
# cfg = Config.fromfile('/home/sbhandari/seg/mmdetection/configs/vit/vitdet_base.py')


# # Modify dataset classes and color
cfg.classes=('stenosis', )
cfg.metainfo = {
    'classes': ('stenosis', ),
}


# Modify dataset type and path according to your dataset
cfg.data_root = 'data'

cfg.train_dataloader.dataset.ann_file = 'annotations/train.json'#path for ann_file for train dataset
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'images'#trainset images
cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.train_dataloader.batch_size = 4
cfg.train_dataloader.num_workers = 4

cfg.val_dataloader.dataset.ann_file = 'annotations/valid.json'#ann_file for validation set
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'images' # valid set images
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.batch_size = 1
cfg.val_dataloader.num_workers = 4

cfg.test_dataloader = cfg.val_dataloader



# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'/'+ 'annotations/valid.json'

cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1


# Set up working dir to save files and logs.
cfg.work_dir = './my_results'

cfg.model.backbone.frozen_stages = 0

# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 1

# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 50

#config for saving best model
cfg.default_hooks.checkpoint.save_best = 'coco/segm_mAP_50'
cfg.default_hooks.checkpoint.rule = 'greater'




set_random_seed(0)


#logging info into visbackend and log file
cfg.default_hooks.logger.interval = 250

#if manually downloaded the model use this

cfg.load_from = 'model.pth'


# build the runner from config
runner = Runner.from_cfg(cfg)

# start training
runner.train()
