_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/coco_custom.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_custom.py'
]

model = dict(
    pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
