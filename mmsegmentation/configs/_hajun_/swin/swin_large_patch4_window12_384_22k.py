checkpoint_file = '../pretrain/swin_large_patch4_window12_384_22k.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48], 
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0., #drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=11),
    auxiliary_head=dict(in_channels=768, num_classes=11))