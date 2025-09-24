import dataloader.data_utils as data_utils
from models.ConCM.base import Trainer
import models.ConCM.Network as ConCM_Net
import torch
import torch.nn as nn
import utils
import os
from utils import load_model
import numpy as np
import datetime
import time







class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.args=data_utils.set_up_datasets(self.args)
        self.set_save_path()
        self.result_list = [self.args]

        self.model = ConCM_Net.ConCM(args)
        self.model = nn.DataParallel(self.model, list(range(args.num_gpu)))
        self.model = self.model.cuda()



    def load_pretrain(self):
        model_path = self.args.pretrain_path + '/' + self.args.backbone+'_'+self.args.dataset + '_model_pretrain.pth'

        if os.path.exists(model_path):
            # only backbone and projector
            best_model_dict = {}
            state_dict = torch.load(model_path, weights_only=True)
            for k, v in state_dict.items():
                if "feature_extractor" in k:
                    best_model_dict[k] = v
                if "projector" in k:
                    best_model_dict[k] = v
            self.model.module.load_state_dict(best_model_dict, strict=False)




    def train_base(self):
        # data
        train_set, test_set = data_utils.get_base_dataloader(self.args)
        # train/load
        model_path = self.args.save_path + '/' + self.args.backbone + '_' + self.args.dataset + '_session_' + str(0) + '.pth'
        if os.path.exists(model_path) and self.args.load_mode=="load":          #load
            self.model = load_model(self.args, model_path,0)
        else:                                                                   #train
            self.model.module.base_knowledge_acquire(train_set)
            ProtoComNet_path = self.args.save_path + '/' + self.args.backbone + '_' + self.args.dataset + '_ProtoComNet' + '.pth'
            if os.path.exists(ProtoComNet_path) and self.args.load_mode=="load":
                self.model.module.ProtoComNet.load_state_dict(torch.load(ProtoComNet_path, weights_only=True))
            else:
                if self.args.protonet_meta_mode=="epoch":
                    self.model.module.train_ProtoComNet_task(train_set)
                elif self.args.protonet_meta_mode=="task":
                    self.model.module.train_ProtoComNet_task(train_set)

            self.model.module.train_Projector(train_set,test_set)
            self.model = load_model(self.args, model_path, 0)

        # base test
        _,acc_base,_,acc_all,output_string=self.model.module.model_test(test_set,0)
        self.result_list.append(output_string)
        self.trlog['max_base_acc'][0] = float('%.2f' % acc_base)
        self.trlog['max_acc'][0] = float('%.2f' % acc_all)

    def train(self):
        t_start_time = time.time()
        torch.cuda.empty_cache()
        self.choose_ablation_mode()
        self.load_pretrain()
        self.train_base()
        utils.set_seed(self.args.rand_seed+1)
        for session in range(1, self.args.sessions):
            print(f"Increment session:{session}")
            if session>self.args.sessions/2:
                self.args.base_weight=self.args.base_weight_half
            train_set = data_utils.get_new_joint_dataloader(self.args, session)
            novel_set, test_set = data_utils.get_new_dataloader(self.args, session)
            # Increment train
            if self.args.is_Frozen == False:
                self.model.module.increment_update(train_set, test_set, novel_set, session)
            else:
                self.model.module.model_frozen(novel_set)
            acc_hm,acc_base,acc_novel,acc_all,output_string=self.model.module.model_test(test_set,session)
            self.trlog['max_hm'][session] = float('%.2f' % acc_hm)
            self.trlog['max_base_acc'][session] = float('%.2f' % acc_base)
            self.trlog['max_novel_acc'][session] = float('%.2f' % acc_novel)
            self.trlog['max_acc'][session] = float('%.2f' % acc_all)
            self.result_list.append(output_string)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        self.trlog['total_time'] = float('%.2f' % total_time)
        self.exit_log(self.result_list)

    def choose_ablation_mode(self):
        if self.args.ablation_mode == 'ConCM':
            self.args.is_MPC = True
            self.args.is_AUG = True
            self.args.is_RepeatSample = True
            self.args.is_DSM = True
            self.args.is_ContLoss = True
            self.args.is_Frozen = False
            if self.args.dataset == "imagenet1000" or self.args.dataset == "mini_imagenet_500s":
                self.args.is_RepeatSample = False
        if self.args.ablation_mode == 'ConCM_wo_AUG':
            self.args.is_MPC = True
            self.args.is_AUG = False
            self.args.is_RepeatSample = True
            self.args.is_DSM = True
            self.args.is_ContLoss = True
            self.args.is_Frozen = False
            if self.args.dataset == "imagenet1000":
                self.args.is_RepeatSample = False
        elif self.args.ablation_mode == 'only_DSM_w_Aug':
            self.args.is_MPC = False
            self.args.is_AUG = True
            self.args.is_RepeatSample = False
            self.args.is_DSM = True
            self.args.is_ContLoss = True
            self.args.is_Frozen = False
        elif self.args.ablation_mode == 'only_DSM_wo_AUG':
            self.args.is_MPC = False
            self.args.is_AUG = False
            self.args.is_RepeatSample = False
            self.args.is_DSM = True
            self.args.is_ContLoss = True
            self.args.is_Frozen = False
        elif self.args.ablation_mode == 'only_MPC':
            self.args.is_MPC = True
            self.args.is_AUG = False
            self.args.is_RepeatSample = True
            self.args.is_DSM = False
            self.args.is_ContLoss = False
            self.args.is_Frozen = False
        elif self.args.ablation_mode == 'finetune':
            self.args.is_MPC = False
            self.args.is_AUG = False
            self.args.is_RepeatSample = False
            self.args.is_DSM = False
            self.args.is_ContLoss = False
            self.args.is_Frozen = False
        elif self.args.ablation_mode == 'frozen':
            self.args.is_MPC = False
            self.args.is_AUG = False
            self.args.is_RepeatSample = False
            self.args.is_DSM = False
            self.args.is_Frozen = True



    def exit_log(self, result_list):
            del self.trlog['max_hm'][0]

            result_list.append("Top 1 Accuracy: ")
            result_list.append(self.trlog['max_acc'])

            result_list.append("Harmonic Mean: ")
            result_list.append(self.trlog['max_hm'])

            result_list.append("Base Test Accuracy: ")
            result_list.append(self.trlog['max_base_acc'])

            result_list.append("Novel Test Accuracy: ")
            result_list.append(self.trlog['max_novel_acc'])

            average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
            result_list.append("Average Harmonic Mean Accuracy: ")
            result_list.append(average_harmonic_mean)

            average_acc = np.array(self.trlog['max_acc']).mean()
            result_list.append("Average Accuracy: ")
            result_list.append(average_acc)

            performance_decay = self.trlog['max_acc'][0] - self.trlog['max_acc'][-1]
            result_list.append("Performance Decay: ")
            result_list.append(performance_decay)

            total_time = self.trlog['total_time']
            result_list.append("total_time: ")
            result_list.append(total_time)



            print(f"\n\nacc: {self.trlog['max_acc']}")
            print(f"avg_acc: {average_acc:.3f}")
            print(f"hm: {self.trlog['max_hm']}")
            print(f"avg_hm: {average_harmonic_mean:.3f}")
            print(f"pd: {performance_decay:.3f}")
            print(f"base: {self.trlog['max_base_acc']}")
            print(f"novel: {self.trlog['max_novel_acc']}")
            print('Total time used %.3f mins' % total_time)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_path = self.args.ablation_mode+'_results_' + current_time + ".txt"
            utils.save_list_to_txt(os.path.join(self.args.save_path, results_path), result_list)

    def set_save_path(self):

        self.args.pretrain_path = self.args.pretrain_path +'/'+ self.args.dataset
        self.args.save_path = self.args.save_path+'/'+self.args.dataset

        utils.ensure_path(self.args.pretrain_path)
        utils.ensure_path(self.args.save_path)