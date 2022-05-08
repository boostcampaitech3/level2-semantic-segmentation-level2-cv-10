# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # mlflow
        dict(
            type='MlflowLoggerHook',
            exp_name='',
        ),
        # wandb
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='', # lv2-p-3, semantic_segmentation
                entity='', # seungri0826, snow-man
                name=''
            ),
        )
    ])

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
