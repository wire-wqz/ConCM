import argparse
import importlib
from utils import set_gpu, set_seed

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # # miniiamgenet
    parser.add_argument('-project', type=str, default="ConCM")
    parser.add_argument('-dataset', type=str, default="mini_imagenet",choices=['mini_imagenet', 'cub200','imagenet1000'])
    parser.add_argument('-dataroot', type=str, default="/root/autodl-fs/dataset" )
    parser.add_argument('-pretrain_path', type=str, default='./params')
    parser.add_argument('-save_path', type=str, default='./checkpoints')


    parser.add_argument('-num_workers', type=int, default=4)

    parser.add_argument('-pretrain_batch', type=int, default=128)
    parser.add_argument('-pretrain_lr', type=float, default=0.1)
    parser.add_argument('-pretrain_epochs', type=int, default=200)
    parser.add_argument('-pretrain_warmup_epochs', type=int, default=10)

    parser.add_argument('-protonet_batch', type=int, default=128)
    parser.add_argument('-protonet_lr', type=float, default=1)
    parser.add_argument('-protonet_epochs', type=int, default=200)
    parser.add_argument('-protonet_warmup_epochs', type=int, default=10)
    parser.add_argument('-protonet_shot', type=int, default=1)
    parser.add_argument('-protonet_task_num', type=int, default=15000)


    parser.add_argument('-align_batch', type=int, default=32)
    parser.add_argument('-align_lr', type=float, default=0.01)
    parser.add_argument('-align_epochs', type=int, default=100)
    parser.add_argument('-align_warmup_epochs', type=int, default=0)

    parser.add_argument('-increment_batch', type=int, default=32)
    parser.add_argument('-increment_lr', type=float, default=0.01)
    parser.add_argument('-increment_epochs', type=int, default=25)
    parser.add_argument('-increment_warmup_epochs', type=int, default=0)

    parser.add_argument('-test_batch', type=int, default=100)

    parser.add_argument('-sample_num_base', type=int, default=100)
    parser.add_argument('-sample_num_novel', type=int, default=50)
    parser.add_argument('-replay_num', type=int, default=-1)

    parser.add_argument('-novel_weight', type=float, default=2)
    parser.add_argument('-base_weight', type=float, default=0.05)
    parser.add_argument('-base_weight_half', type=float, default=0.5)

    parser.add_argument('-calibration_alpha', type=float, default=0.6)
    parser.add_argument('-calibration_gamma', type=float, default=0.6)
    parser.add_argument('-calibration_t', type=float, default=16)

    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('-rand_seed', type=int, default=1)
    parser.add_argument('-contrastive_loss_mode', type=str, default="only_novel",choices=["only_novel", 'all'])
    parser.add_argument('-plt_cm', action='store_true', default=0)
    parser.add_argument('-load_mode', type=str, default="load", choices=['load', 'train'])


    # cub200
    # # 数据集
    # parser.add_argument('-project', type=str, default="ConCM")
    # parser.add_argument('-dataset', type=str, default="cub200",choices=['mini_imagenet', 'cub200','imagenet1000'])
    # parser.add_argument('-dataroot', type=str, default="C:/Users/87902/Desktop/常用数据集" )  #"/root/autodl-fs/dataset" "C:/Users/87902/Desktop/常用数据集" /root/autodl-tmp
    # parser.add_argument('-pretrain_path', type=str, default='./pretrain')
    # parser.add_argument('-save_path', type=str, default='./model_save')
    #
    # parser.add_argument('-num_workers', type=int, default=0)
    #
    # # 预训练
    # parser.add_argument('-pretrain_batch', type=int, default=128)
    # parser.add_argument('-pretrain_lr', type=float, default=0.002)
    # parser.add_argument('-pretrain_epochs', type=int, default=500)
    # parser.add_argument('-pretrain_warmup_epochs', type=int, default=10)
    #
    # # 原型补全网络训练
    # # parser.add_argument('-protonet_batch', type=int, default=256)
    # parser.add_argument('-protonet_lr', type=float, default=1)
    # parser.add_argument('-protonet_epochs', type=int, default=200)
    # parser.add_argument('-protonet_warmup_epochs', type=int, default=0)
    # parser.add_argument('-protonet_shot', type=int, default=5)
    #
    # # 几何结构优化参数
    # parser.add_argument('-align_batch', type=int, default=32)
    # parser.add_argument('-align_lr', type=float, default=0.001)
    # parser.add_argument('-align_epochs', type=int, default=100)
    # parser.add_argument('-align_warmup_epochs', type=int, default=0)
    #
    # # 增量训练
    # parser.add_argument('-increment_batch', type=int, default=32)
    # parser.add_argument('-increment_lr', type=float, default=0.001)
    # parser.add_argument('-increment_epochs', type=int, default=20)
    # parser.add_argument('-increment_warmup_epochs', type=int, default=0)
    #
    # # 测试
    # parser.add_argument('-test_batch', type=int, default=30)
    #
    # # 采样数量
    # parser.add_argument('-sample_num_base', type=int, default=100)
    # parser.add_argument('-sample_num_novel', type=int, default=50)
    # parser.add_argument('-replay_num', type=int, default=-1)
    #
    # # 交叉熵新类基类权重
    # parser.add_argument('-novel_weight', type=float, default=0.5)
    # parser.add_argument('-base_weight', type=float, default=0.5)
    #
    # # 校准权重
    # parser.add_argument('-calibration_alpha', type=float, default=0.75)
    # parser.add_argument('-calibration_gamma', type=float, default=0.6)
    # parser.add_argument('-calibration_t', type=float, default=16)
    #
    # # 其他
    # parser.add_argument('-gpu', type=str, default="0")
    # parser.add_argument('-rand_seed', type=int, default=1)
    #
    # parser.add_argument('-contrastive_loss_mode', type=str, default="only_novel")
    # parser.add_argument('-plt_cm', action='store_true', default=0)
    # parser.add_argument('-load_mode', type=str, default="load", choices=['load', 'train'])

    return parser



if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    args.num_gpu = set_gpu(args)
    set_seed(args.rand_seed)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()








