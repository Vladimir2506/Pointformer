_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py', '../_base_/models/votenet.py', '../_base_/default_runtime.py'
]
data=dict(
    samples_per_gpu=5,
    workers_per_gpu=5
)
model = dict(
    backbone=dict(
        _delete_=True,
        type='Pointformer',
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(16, 16, 16, 16),
        basic_channels=64,
        fp_channels=((256, 256), (256, 256)),
        num_heads=4,
        num_layers=2,
        ratios=(1, 1, 1, 1),
        use_lin_enc=False,
        cloud_points=20000,
        norm_cfg=dict(type='BN2d'),
        use_decoder=(False, False, False, False),
        global_drop=0.1, 
        decoder_drop=0.0,
        prenorm=False
    ),
    bbox_head=dict(
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
                [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
                [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
                [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
                [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
            ]
        )
    )
)
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=5e-2)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32], gamma=0.3)
total_epochs = 36
