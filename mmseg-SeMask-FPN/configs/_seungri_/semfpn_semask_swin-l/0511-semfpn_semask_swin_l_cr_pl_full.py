_base_ = [
    "../_base_/models/semfpn_semask_swin.py",
    "../_base_/datasets/coco_custom_full_pl.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_custom.py",
]
model = dict(
    pretrained="pretrain/swin_large_patch4_window12_384_22k.pth",
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        num_cls=11,
        sem_window_size=12,
        num_sem_blocks=[1, 1, 1, 1],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=11,
        sampler=dict(type="OHEMPixelSampler", min_kept=100000),
        cate_w=0.4,
    ),
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

# learning policy
lr_config = dict(
    _delete_=True,
    policy="CosineRestart",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    min_lr=1e-8,
    periods=[20, 20, 10],
    restart_weights=[1, 0.5, 0.15],
    by_epoch=True,
)

# By default, models are trained on 2 GPUs with 8 images per GPU
data = dict(samples_per_gpu=4)

# fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
# fp16 = dict()

runner = dict(type="EpochBasedRunner", max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=8)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        # wandb
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="semantic_segmentation",
                entity="snow-man",
                name="sr_semfpn_semask_swin-l_cr_pl_full",
            ),
        ),
    ],
)
