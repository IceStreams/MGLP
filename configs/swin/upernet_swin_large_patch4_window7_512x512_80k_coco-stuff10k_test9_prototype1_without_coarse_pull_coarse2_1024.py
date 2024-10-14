EVAL_STEP = 4000
TRAIN_MODE = True

PROTO_SVAE_PATH = "/home/gaokuiliang/fine-grained/mmsegmentation-0.30.0/cocostuff_proto_save_swin_test9_proto1_without_coarse_pull"
PROTO_LOAD_PATH = "/home/gaokuiliang/fine-grained/mmsegmentation-0.30.0/cocostuff_proto_save_swin_test9_proto1_without_coarse_pull/40000_proto_1_512.pt"
DIMENSION_PROTOTYPE = 448
NUM_PROTOTYPES = 1

PROTO_SVAE_PATH_COARSE = "/home/gaokuiliang/fine-grained/mmsegmentation-0.30.0/cocostuff_proto_save_swin_test9_proto1_without_coarse_pull"
PROTO_LOAD_PATH_COARSE = "/home/gaokuiliang/fine-grained/mmsegmentation-0.30.0/cocostuff_proto_save_swin_test9_proto1_without_coarse_pull/40000_proto_1_512_coarse.pt"
DIMENSION_PROTOTYPE_COARSE = 448
NUM_PROTOTYPES_COARSE = 1

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='EncoderDecoder_Test9',
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead_Test9_without_coarse_pull_coarse2',

        num_classes=171,
        num_classes_coarse=27,

        train_mode=TRAIN_MODE,
        eval_step=EVAL_STEP,

        proto_save_path=PROTO_SVAE_PATH,
        proto_load_path=PROTO_LOAD_PATH,
        dimension_prototype=DIMENSION_PROTOTYPE,
        num_prototype=NUM_PROTOTYPES,

        proto_load_path_coarse=PROTO_LOAD_PATH_COARSE,
        proto_save_path_coarse=PROTO_SVAE_PATH_COARSE,
        num_prototype_coarse=NUM_PROTOTYPES_COARSE,
        dimension_prototype_coarse=DIMENSION_PROTOTYPE_COARSE,

        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=DIMENSION_PROTOTYPE,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='PixelPrototypeCELoss')
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=171,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',
                  hiera={'0': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38], '1': [24, 25, 26, 27, 28],
                         '2': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                         '3': [9, 10, 11, 12, 13], '4': [1, 2, 3, 4, 5, 6, 7, 8], '5': [0],
                         '6': [73, 74, 75, 76, 77, 78, 79],
                         '7': [68, 69, 70, 71, 72],
                         '8': [62, 63, 64, 65, 66, 67], '9': [56, 57, 58, 59, 60, 61],
                         '10': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
                         '11': [39, 40, 41, 42, 43, 44, 45], '12': [108, 136, 143, 166, 167],
                         '13': [99, 113, 114, 124, 128, 132, 133, 135, 137, 142, 147],
                         '14': [115, 123, 138, 148, 150, 170], '15': [94, 145],
                         '16': [82, 85, 107, 112, 117, 122, 130, 151, 157],
                         '17': [87, 101, 126, 134, 152], '18': [83, 84, 116, 139, 146, 154], '19': [109, 110, 141, 158],
                         '20': [80, 81, 92, 93, 97, 119, 125, 129, 140, 155, 156],
                         '21': [86, 95, 96, 98, 100, 111, 118, 121, 144, 149, 153],
                         '22': [168, 169], '23': [89, 102, 103, 104, 105, 106], '24': [90, 91],
                         '25': [159, 160, 161, 162, 163, 164, 165],
                         '26': [88, 120, 127, 131]}
                  ))




# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = '/home/gaokuiliang/fine-grained/downloads/cocostuff-10k'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        img_dir='images/train2014',
        ann_dir='annotations/train2014',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        img_dir='images/test2014',
        ann_dir='annotations/test2014',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        img_dir='images/test2014',
        ann_dir='annotations/test2014',
        pipeline=test_pipeline))


# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True




# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

# fp16 placeholder
fp16 = dict()
# learning policy
lr_config = dict(
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=EVAL_STEP*10+1)
checkpoint_config = dict(by_epoch=False, interval=EVAL_STEP)
evaluation = dict(interval=EVAL_STEP, metric='mIoU', pre_eval=True)