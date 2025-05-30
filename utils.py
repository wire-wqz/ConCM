import os
import random
import time
import numpy as np
import torch
import models.ConCM.Network as ConCM_Net
import matplotlib.pyplot as plt
import logging
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        logging.info('create folder:', path)
        os.makedirs(path)

def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def load_model(args,model_path,session):
    model = ConCM_Net.ConCM(args)
    model = nn.DataParallel(model, list(range(args.num_gpu)))
    model = model.cuda()
    state_dict = torch.load(model_path,weights_only=True)
    model.module.load_state_dict(state_dict,strict=False)
    for i in range(1,session+1):
        new_fc_name='prototype_classifier.classifiers.'+str(i)+'.weight'
        novel_rv=state_dict[new_fc_name]
        new_fc = nn.Linear(novel_rv.shape[1], novel_rv.shape[0], bias=False).cuda()
        new_fc.weight.data.copy_(novel_rv)
        model.module.prototype_classifier.classifiers.append(new_fc.cuda())
    model.module.base_knowledge_acquire()
    return model



class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        if x is None:   # Skipping addition of empty items
            return
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():      
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def Confusion_Matrix(true_label_list, predicted_label_list, cmap = "jets"):
    cm = confusion_matrix(true_label_list, predicted_label_list)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, fmt="d", cmap=cmap, xticklabels=np.unique(true_label_list),
                yticklabels=np.unique(true_label_list))

    tick_interval = 25
    xticks = np.arange(0, cm.shape[1], tick_interval)
    yticks = np.arange(0, cm.shape[0], tick_interval)

    plt.xticks(ticks=xticks, labels=xticks, rotation=0, fontsize=24, fontname='Times New Roman')
    plt.yticks(ticks=yticks, labels=yticks, fontsize=24, fontname='Times New Roman')
    plt.xlabel('Predicted Labels', fontsize=26, fontname='Times New Roman')
    plt.ylabel('True Labels', fontsize=26, fontname='Times New Roman')
    plt.show()


def cal_SMD(prototypes,rv):
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    rv = F.normalize(rv, p=2, dim=-1)
    cos_sim = (prototypes * rv).sum(dim=-1)  # shape: (N,)
    avg_cos_sim = cos_sim.mean()
    return avg_cos_sim

def cal_BER(true_label_list,predicted_label_list,base_class):
    binary_true = (true_label_list >= base_class).astype(int)
    binary_pred = (predicted_label_list >= base_class).astype(int)

    cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    fpr = FP / (FP + TN) * 100 if (FP + TN) > 0 else 0.0
    fnr = FN / (TP + FN) * 100 if (TP + FN) > 0 else 0.0
    ber=(fpr+fnr)/2
    return fpr,fnr,ber




