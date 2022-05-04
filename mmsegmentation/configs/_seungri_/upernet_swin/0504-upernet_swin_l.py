_base_ = [
    "../_base_/models/upernet_swin.py",
    "../_base_/datasets/coco_custom.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_custom.py",
]

checkpoint_file = "pretrain/swin_large_patch4_window12_384_22k.pth"  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.0,  # drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=11),
    auxiliary_head=dict(in_channels=768, num_classes=11),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,  # lr=0.000006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,  # warmup_ratio=1e-8,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

# fp16 settings
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
# fp16 placeholder
fp16 = dict()

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        # mlflow
        dict(
            type="MlflowLoggerHook",
            exp_name="0504-upernet_swin-l",
        ),
        # wandb
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="semantic_segmentation",
                entity="snow-man",
                name="sr_upernet_swin-l",
            ),
        ),
    ],
)
