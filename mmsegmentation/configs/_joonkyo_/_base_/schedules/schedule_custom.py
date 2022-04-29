lr = 0.0001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# runtime settings
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=5)
evaluation = dict(metric='mIoU', save_best='mIoU')