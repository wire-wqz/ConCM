python "train.py" \
    -project ConCM \
    -dataset cub200 \
    -dataroot "your_root" \
    -pretrain_path ./params \
    -save_path ./checkpoints \
    -backbone 'resnet18' \
    -protonet_batch 32 \
    -protonet_shot 5 \
    -protonet_meta_mode "task" \
    -protonet_task_num 10000 \
    -align_epochs 100 \
    -align_lr 1e-3  \
    -increment_epochs 20 \
    -increment_lr 1e-3  \
    -test_batch 30 \
    -replay_num -1 \
    -novel_weight 0.5 \
    -base_weight 0.5 \
    -base_weight_half 0.5 \
    -calibration_alpha 0.75 \
    -gpu 0 \
    -contrastive_loss_mode only_novel\
    -load_mode train \



