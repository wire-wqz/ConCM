import numpy as np

def set_up_datasets(args):
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9


    args.Dataset=Dataset

    if args.replay_num == -1:
        args.replay_num = args.shot
    assert args.replay_num <= args.shot, "Exemplars count cannot be greater than the number of shots in your few shot data"

    return args


def get_base_dataloader(args, dino_transform = None):
    #txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index,base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    return trainset, testset

def get_new_dataloader(args, session, dino_transform = None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,index_path=txt_path)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path)


    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,index=class_new)

    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,index=class_new)


    return trainset, testset

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

def get_new_joint_dataloader(args, session):
    # Data replay (only incremental data is replayed)
    if args.dataset == 'cub200':
        txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path)
        for inter_ix in range(1, session):
            txt_path = "data/index_list/" + args.dataset + "/session_" + str(inter_ix + 1) + '.txt'
            # Get data from current index
            inter_set = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path)
            if args.replay_num != args.shot:
                inter_targets = np.array(inter_set.targets)
                for i in np.unique(inter_targets):
                    ixs = np.where(inter_targets == i)[0]
                    selected_ixs = list(ixs[:args.replay_num])
                    for j in selected_ixs:
                        trainset.data.append(inter_set.data[j])
                        trainset.targets.append(inter_set.targets[j])
            else:
                trainset.data.extend(inter_set.data)
                trainset.targets.extend(inter_set.targets)

    if args.dataset == 'mini_imagenet':
        txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path)
        for inter_ix in range(1, session):
            txt_path = "data/index_list/" + args.dataset + "/session_" + str(inter_ix + 1) + '.txt'
            # Get data from current index
            inter_set = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path)
            if args.replay_num != args.shot:
                inter_targets = np.array(inter_set.targets)
                for i in np.unique(inter_targets):
                    ixs = np.where(inter_targets == i)[0]
                    selected_ixs = list(ixs[:args.replay_num])
                    for j in selected_ixs:
                        trainset.data.append(inter_set.data[j])
                        trainset.targets.append(inter_set.targets[j])
            else:
                trainset.data.extend(inter_set.data)
                trainset.targets.extend(inter_set.targets)

    return trainset

