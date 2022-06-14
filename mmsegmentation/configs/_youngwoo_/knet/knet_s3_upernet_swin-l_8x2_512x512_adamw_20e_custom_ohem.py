_base_ = 'knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_custom.py'

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'  # noqa
# model settings
model = dict(
    pretrained=checkpoint_file,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[192, 384, 768, 1536])),
    auxiliary_head=dict(in_channels=768))

# In K-Net implementation we use batch size 2 per GPU as default
data = dict(samples_per_gpu=2)

# === optimizer&scheduler
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    min_lr_ratio=5e-6)

# === runtime config
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=20)
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
            name='yw_knet_upernet_swin-l_cosanlg_ce-dice_ohem'
        ),
        by_epoch=True,
        interval=1
        )
    ])

# === sampler
sampler = dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000)