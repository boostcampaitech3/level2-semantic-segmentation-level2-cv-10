_base_ = [
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/models/upernet_convnext.py',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/datasets/custom.py', 
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/default_runtime_fp16.py',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='xlarge',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048],
        num_classes=11,
    ),
    auxiliary_head=dict(in_channels=1024, num_classes=11),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

# === optimizer&scheduler
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.00008,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    min_lr_ratio=5e-6)

# === runtime config
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(_delete_=True, max_keep_ckpts=2, by_epoch=True, interval=1)

evaluation = dict(interval=1, metric='mIoU', pre_eval=True, classwise=True, save_best='mIoU', by_epoch=True)

# === logger
log_config = dict(
    # interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False, interval=100),
        # wandb
        dict(type='WandbLoggerHook', init_kwargs=dict(
            entity='snow-man',
            project='semantic_segmentation',
            name='yw_upernet_convnextXL'
        ),
        by_epoch=True,
        interval=1
        )
    ])
