_base_ = [
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/models/deeplabv3_unet_s5-d16.py', 
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/datasets/custom.py',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/default_runtime.py', 
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=11),
    auxiliary_head=dict(num_classes=11),
    test_cfg=dict(crop_size=(512, 512), stride=(340, 340))
)