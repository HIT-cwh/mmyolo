_base_ = './yolov6_s_syncbn_fast_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        head_module=dict(type='YOLOv6HeadModuleWithDFL', reg_max=16),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.5 / 4)))
