_base_ = [
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/models/deeplabv3plus_r50-d8.py',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/datasets/custom.py', 
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/default_runtime.py',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-10/mmsegmentation/configs/_youngwoo_/_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(
        num_classes=11,
        loss_decode=dict(type='FocalLoss', loss_name='loss_focal', use_sigmoid=True, loss_weight=1.0)
    ), 
    auxiliary_head=dict(
        num_classes=11,
        loss_decode=dict(type='FocalLoss', loss_name='loss_focal', use_sigmoid=True, loss_weight=0.4)
    ))

# evaluation config 지정
evaluation = dict(metric='mIoU', classwise=True, save_best='mIoU')