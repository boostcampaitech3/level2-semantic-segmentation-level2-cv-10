lr = 0.0001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=0.005)
# optimizer_config = dict(grad_clip=None)
# runtime settings
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=0.1,
    min_lr_ratio=1e-6)

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5)
evaluation = dict(metric='mIoU', save_best='mIoU')