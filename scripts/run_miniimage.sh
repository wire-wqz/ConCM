python "train.py" \
    -project ConCM \
    -dataset mini_imagenet \
    -dataroot "your_root" \
    -pretrain_path ./params \
    -save_path ./checkpoints \
    -backbone 'resnet18' \
    -pretrain_batch 128 \
    -protonet_batch 128 \
    -protonet_meta_mode "task" \
    -protonet_shot 5 \
    -protonet_task_num 10000 \
    -align_epochs 10 \
    -align_lr 1e-2  \
    -increment_epochs 10 \
    -increment_lr 1e-2  \
    -test_batch 100 \
    -replay_num -1 \
    -novel_weight 2 \
    -base_weight 0.05 \
    -base_weight_half 0.05 \
    -calibration_alpha 0.6 \
    -gpu 0 \
    -contrastive_loss_mode only_novel \
    -load_mode train \
    -ablation_mode "ConCM"\













