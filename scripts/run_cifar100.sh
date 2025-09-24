python "train.py" \
    -project ConCM \
    -dataset cifar100 \
    -dataroot "your_root" \
    -pretrain_path ./params \
    -save_path ./checkpoints \
    -backbone 'resnet12' \
    -pretrain_epochs 2000 \
    -protonet_batch 128 \
    -protonet_meta_mode "task" \
    -protonet_shot 5 \
    -protonet_task_num 10000 \
    -align_epochs 100 \
    -align_lr 1e-2  \
    -increment_epochs 20 \
    -increment_lr 2e-2  \
    -test_batch 100 \
    -replay_num -1 \
    -novel_weight 2 \
    -base_weight 0.05 \
    -base_weight_half 0.5 \
    -calibration_alpha 0.6 \
    -gpu 0 \
    -contrastive_loss_mode only_novel\
    -load_mode load \
    -ablation_mode ConCM \


# onlY_DSM
    #-base_weight_half 0.05 \
    #-increment_epochs 20 \

    # baseline/only_MPC 原因：没有对比loss
#     -align_epochs 30 \
#    -novel_weight 0.5 \
#    -base_weight 0.5 \
#    -base_weight_half 0.5 \
#    -increment_epochs 30 \















