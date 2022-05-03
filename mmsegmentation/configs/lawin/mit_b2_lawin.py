_base_ = [

]
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/home/ubuntu/mmsegmentation/pretrain/mit_b2.pth',
    backbone=dict(type='mit_b2', style='pytorch'),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    decode_head=dict(
        type='LawinHead',
        in_channels=[64, 128, 320, 512],
        embed_dim=512,
        use_scale=True,
        reduction=2,
        channels=512,
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))