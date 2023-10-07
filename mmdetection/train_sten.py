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

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


cfg = Config.fromfile('/home/sbhandari/seg/mmdetection/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco copy.py')
# cfg = Config.fromfile('/home/sbhandari/seg/mmdetection/configs/vit/vitdet_base.py')


# # Modify dataset classes and color
cfg.classes=('stenosis', )
cfg.metainfo = {
    'classes': ('stenosis', ),
}


# Modify dataset type and path
cfg.data_root = '/home/sbhandari/combined_data'

cfg.train_dataloader.dataset.ann_file = 'annotations/train.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'images'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.train_dataloader.batch_size = 4
cfg.train_dataloader.num_workers = 4

cfg.val_dataloader.dataset.ann_file = 'annotations/valid.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'images'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.batch_size = 1
cfg.val_dataloader.num_workers = 4

cfg.test_dataloader = cfg.val_dataloader
# cfg.train_cfg.max_epochs = 36


# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'/'+ 'annotations/valid.json'
# cfg.test_evaluator.ann_file = cfg.data_root+'/'+ 'sten_val/annotations/test.json'
# cfg.test_evaluator = cfg.val_evaluator
#for mask-rcnn based
# Modify num classes of the model in box head and mask head 
#for sten 26 ,for seg 25
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1
# cfg.model.roi_head.bbox_head.loss_bbox =dict(type='L1Loss', loss_weight=2.0)
# cfg.model.roi_head.mask_head.loss_mask=dict(
#                 type='CrossEntropyLoss', use_mask=True, loss_weight=2.0)
# #for retinanet
# cfg.model.bbox_head.num_classes = 1
# cfg.model.mask_head.num_classes = 26



# Set up working dir to save files and logs.
cfg.work_dir = './my_results'

cfg.model.backbone.frozen_stages = 0

# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 50
cfg.default_hooks.checkpoint.save_best = 'coco/segm_mAP_50'
cfg.default_hooks.checkpoint.rule = 'greater'

# Set seed thus the results are more reproducible
# cfg.seed = 0
# set_random_seed(0, deterministic=True)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# We can also use tensorboard to log the training process
# cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})



#logging info into visbackend and log file
cfg.default_hooks.logger.interval = 250

#if manually downloaded the model use this
# cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'
cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth'
# cfg.resume = True

# build the runner from config
runner = Runner.from_cfg(cfg)

# start training
runner.train()
