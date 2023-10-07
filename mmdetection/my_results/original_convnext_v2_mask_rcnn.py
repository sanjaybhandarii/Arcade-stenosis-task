model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=0.0,
        gap_before_final_norm=False,
        use_grn=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth',
            prefix='backbone.'),
        frozen_stages=0),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.95),
            max_per_img=100,
            mask_thr_binary=0.5)),
    _scope_='mmdet')
dataset_type = 'CocoDataset'
data_root = '/home/sbhandari/combined_data'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(512, 512),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, _scope_='mmdet'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        _scope_='mmdet'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
        _scope_='mmdet')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmdet'),
    batch_sampler=dict(type='AspectRatioBatchSampler', _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='/home/sbhandari/combined_data',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='RandomResize',
                scale=(512, 512),
                ratio_range=(0.1, 2.0),
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(512, 512),
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(classes=('stenosis', ))))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='/home/sbhandari/combined_data',
        ann_file='annotations/valid.json',
        data_prefix=dict(img='images'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(classes=('stenosis', ))))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='/home/sbhandari/combined_data',
        ann_file='annotations/valid.json',
        data_prefix=dict(img='images'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(classes=('stenosis', ))))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/sbhandari/combined_data/annotations/valid.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    _scope_='mmdet')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    _scope_='mmdet')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=36, val_interval=1, _scope_='mmdet')
val_cfg = dict(type='ValLoop', _scope_='mmdet')
test_cfg = dict(type='TestLoop', _scope_='mmdet')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    _scope_='mmdet',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_rate=0.95, decay_type='layer_wise', num_layers=12))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet'),
    logger=dict(type='LoggerHook', interval=250, _scope_='mmdet'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        _scope_='mmdet',
        max_keep_ckpts=1,
        save_best='coco/segm_mAP_50',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmdet'),
    visualization=dict(type='DetVisualizationHook', _scope_='mmdet'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmdet')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    _scope_='mmdet')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, _scope_='mmdet')
log_level = 'INFO'
load_from = '/home/sbhandari/seg/mmdetection/checkpoints/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947-757ee2dd.pth'
resume = False
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'
image_size = (512, 512)
max_epochs = 36
classes = ('stenosis', )
metainfo = dict(classes=('stenosis', ))
work_dir = './my_results'
