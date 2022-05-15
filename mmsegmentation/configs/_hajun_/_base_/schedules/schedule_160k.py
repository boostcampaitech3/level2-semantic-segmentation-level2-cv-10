lr = 0.0001  # max learning rate
# optimizer
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.005)
#optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=0.1,
    min_lr_ratio=1e-6)# runtime settings

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(max_keep_ckpts=5,by_epoch=True, interval=1)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU', classwise= True, by_epoch=True)
