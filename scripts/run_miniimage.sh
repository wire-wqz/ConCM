python "train.py" \
    -project ConCM \
    -dataset mini_imagenet \
    -dataroot your_root \
    -pretrain_path ./params \
    -save_path ./checkpoints \
    -protonet_batch 128 \
    -protonet_shot 1 \
    -align_epochs 100 \
    -align_lr 1e-2  \
    -increment_epochs 20 \
    -increment_lr 1e-2  \
    -test_batch 100 \
    -replay_num -1 \
    -novel_weight 2 \
    -base_weight 0.05 \
    -base_weight_half 0.5 \
    -calibration_alpha 0.6 \
    -gpu 0 \
    -contrastive_loss_mode only_novel\
    -load_mode train \













