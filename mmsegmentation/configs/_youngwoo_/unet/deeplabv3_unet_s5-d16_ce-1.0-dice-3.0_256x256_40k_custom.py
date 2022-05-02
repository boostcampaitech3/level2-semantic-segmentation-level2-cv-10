_base_ = '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/unet/deeplabv3_unet_s5-d16_256x256_40k_custom.py'
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]))
