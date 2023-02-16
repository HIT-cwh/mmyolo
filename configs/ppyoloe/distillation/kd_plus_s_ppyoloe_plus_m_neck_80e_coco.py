_base_ = '../ppyoloe_plus_s_fast_8xb8-80e_coco.py'

teacher_ckpt = '/mnt/petrelfs/caoweihan.p/ckpt/ppyoloe_plus_m_fast_8xb8-80e_coco_20230104_193132-e4325ada.pth'

norm_cfg = dict(type='BN', affine=False, track_running_stats=False)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py'),
    teacher=dict(
        cfg_path='mmyolo::ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_deliveries=dict(
            batch_atss_assign=dict(
                type='MethodOutputs',
                max_keep_data=1000,
                method_path=
                'mmyolo.models.task_modules.assigners.batch_atss_assigner.BatchATSSAssigner.forward'
            ),
            batch_task_align_assign=dict(
                type='MethodOutputs',
                max_keep_data=1000,
                method_path=
                'mmyolo.models.task_modules.assigners.batch_task_aligned_assigner.BatchTaskAlignedAssigner.forward'
            ),
            # assign=dict(
            #     type='MethodOutputs',
            #     max_keep_data=1000,
            #     method_path='mmyolo.models.dense_heads.ppyoloe_head.PPYOLOEHead.get_assigned_result')
        ),
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0'),
            fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2'),
            # flatten_cls_preds
            loss_cls_input=dict(
                type='ModuleInputs', source='bbox_head.loss_cls'),
            # pred_bboxes_pos, assigned_scores_sum, bbox_weight
            loss_bbox_input=dict(
                type='MethodInputs',
                source='mmyolo.models.losses.iou_loss.IoULoss.forward'),
            # pred_dist_pos
            loss_dfl_input=dict(
                type='MethodInputs',
                source=
                'mmdet.models.losses.gfocal_loss.DistributionFocalLoss.forward'
            ),
            gt_info=dict(
                type='FunctionOutputs',
                source=
                'mmyolo.models.dense_heads.ppyoloe_head.gt_instances_preprocess'
            ),
            batch_img_metas=dict(
                type='MethodInputs',
                source=
                'mmyolo.models.dense_heads.ppyoloe_head.PPYOLOEHead.loss_by_feat'
            )),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0'),
            fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2'),
            # flatten_cls_preds
            loss_cls_input=dict(
                type='ModuleInputs', source='bbox_head.loss_cls'),
            # pred_bboxes_pos, assigned_scores_sum, bbox_weight
            loss_bbox_input=dict(
                type='MethodInputs',
                source='mmyolo.models.losses.iou_loss.IoULoss.forward'),
            # pred_dist_pos
            loss_dfl_input=dict(
                type='MethodInputs',
                source=
                'mmdet.models.losses.gfocal_loss.DistributionFocalLoss.forward'
            ),
            fg_mask_pre_prior=dict(
                type='MethodOutputs',
                source=
                'mmyolo.models.dense_heads.ppyoloe_head.PPYOLOEHead.get_assigned_result'
            )),
        connectors=dict(
            fpn0_s=dict(
                type='ConvModuleConnector',
                in_channel=96,
                out_channel=144,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn0_t=dict(
                type='NormConnector', in_channels=144, norm_cfg=norm_cfg),
            fpn1_s=dict(
                type='ConvModuleConnector',
                in_channel=192,
                out_channel=288,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn1_t=dict(
                type='NormConnector', in_channels=288, norm_cfg=norm_cfg),
            fpn2_s=dict(
                type='ConvModuleConnector',
                in_channel=384,
                out_channel=576,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn2_t=dict(
                type='NormConnector', in_channels=576, norm_cfg=norm_cfg)),
        distill_losses=dict(
            loss_fgd_fpn0=dict(
                type='FGDLoss',
                in_channels=144,
                alpha_fgd=0.00001,
                beta_fgd=0.000005,
                gamma_fgd=0.00001,
                lambda_fgd=0.00000005),
            loss_fgd_fpn1=dict(
                type='FGDLoss',
                in_channels=288,
                alpha_fgd=0.00001,
                beta_fgd=0.000005,
                gamma_fgd=0.00001,
                lambda_fgd=0.00000005),
            loss_fgd_fpn2=dict(
                type='FGDLoss',
                in_channels=576,
                alpha_fgd=0.00001,
                beta_fgd=0.000005,
                gamma_fgd=0.00001,
                lambda_fgd=0.00000005),
            loss_qfl=dict(type='QualityFocalLoss', loss_weight=4.0),
            loss_dfl=dict(type='DistributionFocalLoss', loss_weight=2.0),
            loss_bbox=dict(type='BboxLoss', loss_weight=10.0),
            # todo: loss_weight and tau are different from official ppyoloe repo
            loss_kl=dict(type='MainKDLoss', loss_weight=4.0, tau=1.0),
        ),
        loss_forward_mappings=dict(
            loss_fgd_fpn0=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn0', connector='fpn0_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn0', connector='fpn0_t'),
                gt_info=dict(from_student=True, recorder='gt_info'),
                batch_img_metas=dict(
                    from_student=True, recorder='batch_img_metas',
                    data_idx=4)),
            loss_fgd_fpn1=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn1', connector='fpn1_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn1', connector='fpn1_t'),
                gt_info=dict(from_student=True, recorder='gt_info'),
                batch_img_metas=dict(
                    from_student=True, recorder='batch_img_metas',
                    data_idx=4)),
            loss_fgd_fpn2=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn2', connector='fpn2_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn2', connector='fpn2_t'),
                gt_info=dict(from_student=True, recorder='gt_info'),
                batch_img_metas=dict(
                    from_student=True, recorder='batch_img_metas',
                    data_idx=4)),
            loss_kl=dict(
                mask_positive=dict(
                    from_student=False,
                    recorder='fg_mask_pre_prior',
                    data_idx=2),
                pred_scores=dict(
                    from_student=True, recorder='loss_cls_input', data_idx=0),
                soft_cls=dict(
                    from_student=False, recorder='loss_cls_input', data_idx=0),
            ),
            loss_qfl=dict(
                preds_S=dict(
                    from_student=True, recorder='loss_cls_input', data_idx=0),
                preds_T=dict(
                    from_student=False, recorder='loss_cls_input', data_idx=0),
                num_total_pos=dict(
                    from_student=True, recorder='loss_bbox_input',
                    data_idx=3)),
            loss_dfl=dict(
                preds_S=dict(
                    from_student=True, recorder='loss_dfl_input', data_idx=0),
                preds_T=dict(
                    from_student=False, recorder='loss_dfl_input', data_idx=0),
                weight_targets=dict(
                    from_student=True, recorder='loss_dfl_input', data_idx=2)),
            loss_bbox=dict(
                s_bbox=dict(
                    from_student=True, recorder='loss_bbox_input', data_idx=0),
                t_bbox=dict(
                    from_student=False, recorder='loss_bbox_input',
                    data_idx=0),
                weight_targets=dict(
                    from_student=True, recorder='loss_bbox_input',
                    data_idx=2)),
        )))

find_unused_parameters = True

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4))
