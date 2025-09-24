import utils
import models.supcon as supcon
from models.resnet18 import ResNet18
from models.ConCM.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import  DataLoader
from tqdm import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torchvision.models as torch_models
from models.ConCM.helper import get_eft_matrix,GaussianDataset_cov
import os
import numpy as np
from dataloader.sampler import CategoriesSampler
from models.resnet12 import resnet12_nc



class FeatureExtractor(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        if self.args.backbone=="resnet18":
            if self.args.dataset=='mini_imagenet':
                self.encoder = ResNet18()
                self.encoder.fc = nn.Identity()

            if self.args.dataset=='cub200':
                # The encoder selected is the ResNet18 pre-trained by imagenet.
                self.encoder = torch_models.resnet18(weights=torch_models.ResNet18_Weights.IMAGENET1K_V1)
                self.encoder.fc = nn.Identity()

        elif self.args.backbone=="resnet12":
            self.encoder = resnet12_nc()
            self.encoder.fc = nn.Identity()


    def forward(self, x):
        if self.args.backbone == "resnet18":
            output=self.encoder(x)
        elif self.args.backbone == "resnet12":
            output=self.encoder(x)
        return output


class Projector(nn.Module):
    def __init__(self,encoder_outdim,proj_hidden_dim,proj_output_dim):
        super().__init__()
        self.encoder_outdim=encoder_outdim
        self.proj_hidden_dim=proj_hidden_dim
        self.proj_output_dim=proj_output_dim

        self.projector = nn.Sequential(
            nn.Linear(self.encoder_outdim, self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        x = self.projector(x)
        x=F.normalize(x,p=2,dim=-1)
        return x


class PseudoTargetClassifier(nn.Module):
    def __init__(self, args, proj_output_dim):
        super().__init__()

        self.args = args
        self.num_features = proj_output_dim
        self.base_fc = nn.Linear(self.num_features, self.args.base_class, bias=False)
        init.normal_(self.base_fc.weight, mean=0, std=1)
        self.classifiers = nn.ModuleList([self.base_fc])

    def forward(self, x):
        return self.get_logits(x)

    def get_logits(self, x):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            out = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            out = out / 1
            output.append(out)
        output = torch.cat(output, axis=1)
        return output

    def assign_classifier(self, prototypes,mode="base"):
        if self.args.is_DSM == False:
            raise ValueError(f"need is_DSM == True")

        prototypes = F.normalize(prototypes, p=2, dim=-1)
        # Geometric structure
        rv=get_eft_matrix(prototypes)

        if mode=='base':
            self.classifiers[0].weight.data = rv

        elif mode=='novel':
            num_class = rv.shape[0]
            num_novel=num_class-self.args.base_class
            session=int(num_novel/self.args.way)

            base_rv = rv[0:self.args.base_class]
            novel_rv = rv[-self.args.way:]

            self.classifiers[0].weight.data = base_rv
            for i in range(1,session):
                self.classifiers[i].weight.data = rv[self.args.base_class+self.args.way*(i-1):self.args.base_class+self.args.way*(i)]
            new_fc = nn.Linear(novel_rv.shape[1], novel_rv.shape[0], bias=False).cuda()
            new_fc.weight.data.copy_(novel_rv)
            self.classifiers.append(new_fc.cuda())

        else:
            raise ValueError(f"Invalid mode value: {mode}. Only 'base' and 'novel' is allowed.")

    def init_Fc(self):
        if self.args.is_DSM == True:
            raise ValueError(f"need is_DSM == False")
        new_fc = nn.Linear(self.num_features, self.args.way, bias=False).cuda()
        self.classifiers.append(new_fc.cuda())

    def get_classifier_weights(self, uptil=-1):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            if uptil >= 0 and uptil < i + 1:
                break
            output.append(cls.weight.data)
        return torch.cat(output, axis=0)

    def test_eft(self,vec):
        vec=self.get_classifier_weights()
        sim = F.cosine_similarity(vec[None, :, :], vec[:, None, :], dim=-1)
        print(sim)


class ProtoComNet(nn.Module):
    def __init__(self, args, in_dim=512):
        super(ProtoComNet, self).__init__()

        self.args=args
        self.d=128
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim//2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_dim//2, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=in_dim)
        )

        self.query_feature_w = nn.Linear(in_dim, self.d, bias=False)
        self.key_feature_w = nn.Linear(in_dim, self.d, bias=False)

        self.query_word_w = nn.Linear(300, self.d, bias=False)
        self.key_word_w = nn.Linear(300, self.d, bias=False)

        self.in_dim=in_dim

        # Obtain semantic
        prior_path = './prior/' + self.args.dataset + '_part_prior_train.pickle'
        with open(prior_path, 'rb') as handle:
            part_prior = pickle.load(handle)

        attribute_vectors=part_prior["attribute_vectors"]
        self.attribute_vectors = F.normalize(attribute_vectors).cuda()
        class_vectors=part_prior["class_vectors"]
        self.class_vectors = F.normalize(class_vectors).cuda()

        self.classid_attributeid_dict=part_prior["classid_attributeid_dict"]


    def forward(self,features,labels,attribute_feature_mean,phase="test",true_proto=[]):
        self.attribute_feature_mean = attribute_feature_mean
        self.n = self.attribute_feature_mean.shape[0]

        if phase not in ["train", "test"]:
            raise ValueError('Invalid phase value. It must be either "train" or "test".')

        outputs = []
        targets = []
        for i in range(features.shape[0]):
            input_proto = features[i:i + 1, :]
            this_attributeid = self.classid_attributeid_dict[labels[i].item()]

            if phase=="train":
                targets.append(true_proto[labels[i]:labels[i]+1,:])

            if this_attributeid==[]:
                ouput_proto = input_proto
                outputs.append(ouput_proto)
            else:
                input_feature = input_proto
                this_class_vectors = self.class_vectors[labels[i]:labels[i] + 1, :]
                input_feature = torch.cat((input_feature, self.attribute_feature_mean), dim=0)
                query_feature = self.query_feature_w(input_proto)
                key_feature = self.key_feature_w(attribute_feature_mean)
                weight_feature = torch.matmul(query_feature, key_feature.transpose(-1, -2)) / math.sqrt(self.d)


                query_word = self.query_word_w(this_class_vectors)
                key_word = self.key_word_w(self.attribute_vectors)
                weight_word = torch.matmul(query_word, key_word.transpose(-1, -2)) / math.sqrt(self.d)
                weight=0.5*weight_word+0.5*weight_feature
                weight=weight.transpose(-1, -2)

                relation = torch.zeros((self.n, 1))
                relation[this_attributeid] = 1
                relation=relation.cuda()
                fuse_weight= weight*relation #final weight
                fuse_weight = torch.cat((torch.tensor([[1.0]], device='cuda:0'), fuse_weight), dim=0)
                non_zero_elements = fuse_weight[fuse_weight != 0]
                softmax_non_zero = F.softmax(non_zero_elements, dim=0)
                fuse_weight[fuse_weight != 0] = softmax_non_zero

                z=self.encoder(input_feature)
                fuse_weight_transposed = torch.transpose(fuse_weight, 0, 1)
                g =torch.mm(fuse_weight_transposed,z)
                ouput_proto = self.decoder(g)
                outputs.append(ouput_proto)

        outputs = torch.cat(outputs, dim=0)
        if phase == "train":
            targets = torch.cat(targets, dim=0)
            return outputs,targets
        else:
            return outputs


class ConCM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args= args

        if self.args.backbone == "resnet18":
            feature_dim = 512
        elif self.args.backbone == "resnet12":
            feature_dim = 640

        if self.args.dataset=='mini_imagenet':
            proj_hidden_dim=2048
            proj_output_dim=128

        elif self.args.dataset=='cub200':
            proj_hidden_dim=2048
            proj_output_dim=256

        elif self.args.dataset == 'cifar100':
            proj_hidden_dim = 2048
            proj_output_dim = 128

        self.feature_extractor = FeatureExtractor(self.args)
        self.projector = Projector(feature_dim,proj_hidden_dim,proj_output_dim)
        self.prototype_classifier = PseudoTargetClassifier(self.args, proj_output_dim)
        self.ProtoComNet=ProtoComNet(self.args,feature_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        projected_features = self.projector(features)
        logic = self.prototype_classifier(projected_features)
        return logic

    def base_knowledge_acquire(self,trainset={}):
        save_path=self.args.save_path+'/'+ self.args.backbone + '_' + self.args.dataset + '_feature_statistics.pt'
        if trainset=={}:
            mode="load"
        else:
            mode=self.args.load_mode
        if os.path.exists(save_path) and mode=="load":
            part = torch.load(save_path,weights_only=True)
            self.base_feature_mean = part['base_feature_mean']
            self.base_feature_cov = part['base_feature_cov']
            self.attribute_feature_mean=part['attribute_feature_mean']
        else:
            prior_path = './prior/' + self.args.dataset + '_part_prior_train.pickle'
            self.eval()
            train_loader = DataLoader(dataset=trainset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.num_workers,pin_memory=True)
            self.base_feature_mean,self.base_feature_cov = self.get_prototypes(train_loader, 'feature')
            with open(prior_path, 'rb') as handle:
                part_prior = pickle.load(handle)
            base_attributeid_classid_dict=part_prior["base_attributeid_classid_dict"]
            self.attribute_feature_mean=self.get_attribute_feature_statistic(train_loader, base_attributeid_classid_dict)

            torch.save({'base_feature_mean': self.base_feature_mean,"base_feature_cov": self.base_feature_cov,"attribute_feature_mean":self.attribute_feature_mean}, save_path)


    def train_ProtoComNet_task(self,trainset):
        sampler = CategoriesSampler(trainset.targets,self.args.protonet_task_num , self.args.base_class, self.args.protonet_shot)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=self.args.num_workers,pin_memory=True)
        save_path = self.args.save_path+ '/' +self.args.backbone + '_' + self.args.dataset + '_ProtoComNet' + '.pth'

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.eval()

        optimizer = torch.optim.SGD([{'params': self.ProtoComNet.parameters()}], lr=self.args.protonet_lr, momentum=0.9, dampening=0.9,weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.protonet_warmup_epochs,max_epochs=self.args.protonet_task_num,warmup_start_lr=3e-05,eta_min=1e-5)
        mse_loss = nn.MSELoss()
        tqdm_gen = tqdm(range(self.args.protonet_task_num), total=self.args.protonet_task_num)
        train_iterator = iter(train_loader)
        for epoch in tqdm_gen:
            running_loss = utils.Averager()
            batch = next(train_iterator)
            images, labels = batch
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.no_grad():
                features = self.feature_extractor(images)
                unique_labels = torch.unique(labels)
                prototypes = []
                for label in unique_labels:
                    label_features = features[labels == label]
                    prototype = label_features.mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
            output_proto, true_proto=self.ProtoComNet(prototypes, unique_labels,self.attribute_feature_mean,phase="train",true_proto=self.base_feature_mean)
            loss = mse_loss(output_proto,true_proto)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss.add(loss.item())
                out_string = f'Epoch: [[{epoch}/{self.args.protonet_task_num}], Loss: {running_loss.item():.4f}'
                tqdm_gen.set_description(out_string)
            scheduler.step()
        torch.save(self.ProtoComNet.state_dict(), save_path)

    def train_ProtoComNet_epoch(self,trainset):
        train_loader = DataLoader(dataset=trainset, batch_size=self.args.protonet_batch, shuffle=True, num_workers=self.args.num_workers,pin_memory=True)
        save_path = self.args.save_path+ '/' + self.args.backbone + '_' + self.args.dataset + '_ProtoComNet' + '.pth'

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.eval()

        optimizer = torch.optim.SGD([{'params': self.ProtoComNet.parameters()}], lr=self.args.protonet_lr, momentum=0.9, dampening=0.9,weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.protonet_warmup_epochs,max_epochs=self.args.protonet_epochs,warmup_start_lr=3e-05,eta_min=1e-5)
        mse_loss = nn.MSELoss()
        tqdm_gen = tqdm(range(self.args.protonet_epochs), total=self.args.protonet_epochs)
        for epoch in tqdm_gen:
            running_loss = utils.Averager()
            optimizer.zero_grad()
            for idx, batch in enumerate(train_loader):
                images, labels = batch
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.no_grad():
                    features = self.feature_extractor(images)
                output_proto, true_proto=self.ProtoComNet(features, labels,self.attribute_feature_mean,phase="train",true_proto=self.base_feature_mean)
                loss = mse_loss(output_proto,true_proto)
                loss.backward()
                if (idx + 1) % self.args.pretrain_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                with torch.no_grad():
                    running_loss.add(loss.item())
                    out_string = f'Epoch: [{epoch + 1}/{self.args.protonet_epochs}],[{idx}/{len(train_loader)}], Loss: {running_loss.item():.4f}'
                    tqdm_gen.set_description(out_string)
            scheduler.step()
        torch.save(self.ProtoComNet.state_dict(), save_path)

    def train_Projector(self, trainset,testset):
        train_loader = DataLoader(dataset=trainset, batch_size=self.args.align_batch, shuffle=True, num_workers=self.args.num_workers,pin_memory=True)
        test_loader = DataLoader(dataset=testset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.num_workers,pin_memory=True)
        save_path = self.args.save_path+'/'+ self.args.backbone + '_' +self.args.dataset + '_session_' + str(0) + '.pth'
        self.eval()

        # Geometric structure
        with torch.no_grad():
            if self.args.is_DSM == True:
                base_proj_proto = self.get_prototypes(train_loader, 'proj_image')
                self.prototype_classifier.assign_classifier(base_proj_proto,'base')
            else:
                self.prototype_classifier.init_Fc()

        if self.args.is_DSM == True:
            optimizer = torch.optim.SGD([{'params': self.projector.parameters()}], lr=self.args.align_lr, momentum=0.9, dampening=0.9, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD([{'params': self.projector.parameters()},{'params': self.prototype_classifier.parameters()}], lr=self.args.align_lr, momentum=0.9,dampening=0.9, weight_decay=1e-4)

        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.align_warmup_epochs,
                                                  max_epochs=self.args.align_epochs,
                                                  warmup_start_lr=3e-05 if self.args.align_warmup_epochs > 0 else self.args.align_lr,
                                                  eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()
        self.eval()

        tqdm_gen = tqdm(range(self.args.align_epochs), total=self.args.align_epochs)
        for epoch in tqdm_gen:
            running_loss = utils.Averager()
            running_acc = utils.Averager()
            for idx, batch in enumerate(train_loader):
                images, labels = batch
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                optimizer.zero_grad()
                feas = self.feature_extractor(images)
                projs = self.projector(feas)
                outputs = self.prototype_classifier(projs)
                # Feature - Geometric Alignment loss
                loss1 = criterion(outputs, labels.long())
                if self.args.is_ContLoss == True:
                    # Feature - Geometric Contrast loss
                    loss2 = self.contrastive_loss(projs, labels, 0)
                    loss = loss1 + loss2
                else:
                    loss = loss1

                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    running_loss.add(loss.item())
                    running_acc.add(utils.count_acc(outputs,labels))
                    out_string = f'Epoch: [{epoch + 1}/{self.args.align_epochs}],[{idx}/{len(train_loader)}], Acc_train: {running_acc.item()*100:.4f}, Loss: {running_loss.item():.4f}'
                    tqdm_gen.set_description(out_string)
            scheduler.step()

        torch.save(self.state_dict(), save_path)

    def increment_update(self,trainset,testset,novelset,session):
        novel_loader = DataLoader(dataset=novelset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.num_workers,pin_memory=True)
        train_loader = DataLoader(dataset=trainset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.num_workers,pin_memory=True)
        test_loader = DataLoader(dataset=testset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.num_workers,pin_memory=True)
        save_path = self.args.save_path + '/' + self.args.backbone + '_' + self.args.dataset + '_session_' + str(session) + '.pth'

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        if not hasattr(self, 'feature_mean'):
            self.feature_mean = self.base_feature_mean

        if not hasattr(self, 'feature_cov'):
            self.feature_cov = self.base_feature_cov

        if self.args.replay_num == self.args.shot:
            novel_feature_mean, novel_feature_cov, all_novel_feature_list, all_novel_label_list = self.get_novel_feature_statistic(train_loader, num_enhance=self.args.num_enhance)
            labels_list = range(self.args.base_class, self.args.base_class + session * self.args.way)
        else:
            _, _, all_novel_feature_list, all_novel_label_list = self.get_novel_feature_statistic(train_loader, num_enhance=self.args.num_enhance)
            novel_feature_mean, novel_feature_cov,_,_ = self.get_novel_feature_statistic(novel_loader,num_enhance=self.args.num_enhance)
            labels_list = range(self.args.base_class + (session-1) * self.args.way, self.args.base_class + session * self.args.way)

        # MPC net
        novel_feature_mean, novel_feature_cov = self.calibration(novel_feature_mean, novel_feature_cov, labels_list)

        if self.args.replay_num == self.args.shot:
            feature_mean = torch.cat((self.base_feature_mean, novel_feature_mean), dim=0)
            feature_cov = torch.cat((self.base_feature_cov, novel_feature_cov), dim=0)
        else:
            feature_mean = torch.cat((self.feature_mean, novel_feature_mean), dim=0)
            feature_cov = torch.cat((self.feature_cov, novel_feature_cov), dim=0)

        dataset = GaussianDataset_cov(feature_mean, feature_cov, self.args.sample_num_base, self.args.sample_num_novel, self.args.base_class, all_novel_feature_list,all_novel_label_list)
        Gtrain_loader = DataLoader(dataset=dataset, batch_size=self.args.increment_batch, shuffle=True, num_workers=self.args.num_workers, pin_memory=True,drop_last=True)

        with torch.no_grad():
            if self.args.is_DSM == True:
                proj_proto = self.get_prototypes(Gtrain_loader, 'proj_feature')
                self.prototype_classifier.assign_classifier(proj_proto, 'novel')
            else:
                self.prototype_classifier.init_Fc()

        if self.args.is_DSM == True:
            optimizer = torch.optim.SGD([{'params': self.projector.parameters()}], lr=self.args.increment_lr, momentum=0.9, dampening=0.9, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD([{'params': self.projector.parameters()},{'params': self.prototype_classifier.parameters()}], lr=self.args.increment_lr, momentum=0.9, dampening=0.9, weight_decay=1e-4)

        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.increment_warmup_epochs,
                                                  max_epochs=self.args.increment_epochs,
                                                  warmup_start_lr=3e-05 if self.args.increment_warmup_epochs > 0 else self.args.increment_lr,
                                                  eta_min=0)
        self.eval()# follow orco

        tqdm_gen = tqdm(range(self.args.increment_epochs), total=self.args.increment_epochs)
        for epoch in tqdm_gen:
            running_loss = utils.Averager()
            running_acc = utils.Averager()

            # Repeated Prototype Augmentation
            if self.args.is_RepeatSample == True:
                if self.args.replay_num == self.args.shot:
                    novel_feature_mean, novel_feature_cov, all_novel_feature_list, all_novel_label_list = self.get_novel_feature_statistic(train_loader, num_enhance=self.args.num_enhance)
                    novel_feature_mean, novel_feature_cov = self.calibration(novel_feature_mean,novel_feature_cov,labels_list)
                    feature_mean = torch.cat((self.base_feature_mean, novel_feature_mean), dim=0)
                    feature_cov=torch.cat((self.base_feature_cov, novel_feature_cov), dim=0)
                    dataset = GaussianDataset_cov(feature_mean, feature_cov, self.args.sample_num_base, self.args.sample_num_novel, self.args.base_class,all_novel_feature_list, all_novel_label_list)
                    Gtrain_loader = DataLoader(dataset=dataset, batch_size=self.args.increment_batch, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
                else:
                    _, _, all_novel_feature_list, all_novel_label_list = self.get_novel_feature_statistic(train_loader,num_enhance=self.args.num_enhance)
                    novel_feature_mean, novel_feature_cov,_,_ = self.get_novel_feature_statistic(novel_loader,num_enhance=self.args.num_enhance)
                    novel_feature_mean, novel_feature_cov = self.calibration(novel_feature_mean, novel_feature_cov, labels_list)
                    feature_mean = torch.cat((self.feature_mean, novel_feature_mean), dim=0)
                    feature_cov = torch.cat((self.feature_cov, novel_feature_cov), dim=0)
                    dataset = GaussianDataset_cov(feature_mean, feature_cov, self.args.sample_num_base,self.args.sample_num_novel, self.args.base_class, all_novel_feature_list,all_novel_label_list)
                    Gtrain_loader = DataLoader(dataset=dataset, batch_size=self.args.increment_batch, shuffle=True,num_workers=self.args.num_workers, pin_memory=True, drop_last=True)

            optimizer.zero_grad()
            for idx, batch in enumerate(Gtrain_loader):
                images, labels = batch
                features_gaussian = images.cuda()
                labels = labels.cuda()
                projs = self.projector(features_gaussian)
                outputs = self.prototype_classifier(projs)
                # Feature - Geometric Alignment loss
                loss1 = self.align_loss(outputs,labels,self.args.base_class)
                if self.args.is_ContLoss == True:
                    # Feature - Geometric Contrast loss
                    loss2=self.contrastive_loss(projs,labels,session)
                    loss=loss1+loss2
                else:
                    loss=loss1
                loss.backward()
                if (idx + 1) % self.args.increment_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                with torch.no_grad():
                    running_loss.add(loss.item())
                    running_acc.add(utils.count_acc(outputs, labels))
                    out_string = f'Epoch: [{epoch + 1}/{self.args.increment_epochs}],[{idx+1}/{len(Gtrain_loader)}], Acc_train: {running_acc.item() * 100:.4f}, Loss: {running_loss.item():.4f}'
                    tqdm_gen.set_description(out_string)
            scheduler.step()
        self.feature_mean = feature_mean
        self.feature_cov=feature_cov


    def model_test(self,testset,session=0):
        testloader = DataLoader(dataset=testset, batch_size=self.args.test_batch, shuffle=False,num_workers=self.args.num_workers, pin_memory=True)
        self.eval()
        acc_all = utils.Averager()
        with torch.no_grad():
            correct_novel=0.0
            correct_base=0.0
            total_novel =0.0
            total_base = 0.0
            predicted_label_list = []
            true_label_list = []
            tqdm_gen=enumerate(testloader)
            for idx, batch in tqdm_gen:
                images, labels = batch
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = self.forward(images)
                _, predicted_label = torch.max(outputs.data, 1)
                true_label_list.append(labels.cpu().numpy())
                predicted_label_list.append(predicted_label.cpu().numpy())
                acc_all.add(utils.count_acc(outputs, labels))
                if session>0:
                    correct_novel += ((labels >= self.args.base_class) & (predicted_label == labels)).sum().item()
                    correct_base += ((labels < self.args.base_class) & (predicted_label == labels)).sum().item()
                    total_novel += (labels >= self.args.base_class).sum().item()
                    total_base += (labels < self.args.base_class).sum().item()
            acc_all = acc_all.item() * 100
            if session > 0:
                acc_base = 100 * correct_base / total_base
                if total_novel != 0:
                    acc_novel = 100 * correct_novel / total_novel
                else:
                    acc_novel = 0
                acc_hm=2 * acc_base * acc_novel / (acc_base + acc_novel)
            else:
                acc_base=acc_all
                acc_novel=0
                acc_hm = 2 * acc_base * acc_novel / (acc_base + acc_novel)

            predicted_label_list = np.concatenate(predicted_label_list)
            true_label_list = np.concatenate(true_label_list)
            if self.args.plt_cm == True:
                utils.Confusion_Matrix(true_label_list, predicted_label_list, cmap="Reds")
            output_string=f'Session {session},acc_hm: {acc_hm:.2f}%, acc_base: {acc_base:.2f}%, acc_novel: {acc_novel:.2f}%, acc: {acc_all:.2f}%'
            print(output_string)
            return acc_hm,acc_base,acc_novel,acc_all,output_string

    def model_frozen(self, novel_set):
        novel_loader = DataLoader(dataset=novel_set, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        self.eval()
        novel_proj_proto = self.get_prototypes(novel_loader, "proj_image")
        new_fc = nn.Linear(novel_proj_proto.shape[1], novel_proj_proto.shape[0], bias=False).cuda()
        new_fc.weight.data.copy_(novel_proj_proto)
        self.prototype_classifier.classifiers.append(new_fc.cuda())


    def get_prototypes(self,train_loader,mode):
        self.eval()
        output_list=[]
        label_list=[]
        with torch.no_grad():
            tqdm_gen = tqdm(enumerate(train_loader), total=len(train_loader))
            for idx, batch in tqdm_gen:
                images, labels = batch
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                if mode=='feature':
                    output = self.feature_extractor(images)
                elif mode=='proj_image':
                    embedding = self.feature_extractor(images)
                    output = self.projector(embedding)
                elif mode=='proj_feature':
                    embedding=images
                    output = self.projector(embedding)
                else:
                    raise ValueError(f"Invalid mode value: {mode}. Only 'feature','proj_image' and 'proj_feature' is allowed.")

                output_list.append(output)
                label_list.append(labels)

            output_list = torch.cat(output_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            label_list=label_list-label_list.min()
            label_count = torch.bincount(label_list)
            num_class=label_list.max()-label_list.min()+1

            one_hot_label = F.one_hot(label_list.long(), num_classes=num_class).float()
            prototypes=torch.matmul(one_hot_label.T,output_list)
            prototypes=prototypes/label_count[:, None]

            if mode == 'feature':
                cov_list=[]
                for class_index in range(num_class):
                    data_index = (label_list == class_index).nonzero()
                    features_this= output_list[data_index.squeeze(-1)]
                    cov_this = torch.std(features_this, dim=0, unbiased=False)
                    cov_list.append(cov_this)
                cov = torch.stack(cov_list, dim=0)
                return prototypes,cov
            else:
                return prototypes

    def get_attribute_feature_statistic(self,train_loader,base_attributeid_classid_dict):
        feature_list = []
        label_list = []
        with torch.no_grad():
            tqdm_gen = tqdm(enumerate(train_loader), total=len(train_loader))
            for idx, batch in tqdm_gen:
                images, labels = batch
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                feature = self.feature_extractor(images)
                feature_list.append(feature)
                label_list.append(labels)

            feature_list = torch.cat(feature_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

        attribute_mean=[]

        for attr_id in base_attributeid_classid_dict.keys():
            attr_this_id=[]
            for class_id in base_attributeid_classid_dict[attr_id]:
                this_id= (label_list == class_id).nonzero(as_tuple=True)[0]
                attr_this_id.extend(this_id)
            attr_this = feature_list[attr_this_id, :]
            mean = torch.mean(attr_this, dim=0).unsqueeze(dim=0)
            attribute_mean.append(mean)

        attribute_mean = torch.cat(attribute_mean, dim=0)

        return attribute_mean

    def get_novel_feature_statistic(self,novel_loader,num_enhance=3):
        self.eval()
        novel_feature_list = []
        novel_label_list = []
        novel_cov = []
        novel_mean = []
        with torch.no_grad():
            for _ in range(num_enhance):
                output_list = []
                label_list = []
                for idx, batch in enumerate(novel_loader):
                    images, labels = [_.cuda() for _ in batch]
                    novel_feature = self.feature_extractor(images)
                    novel_feature_list.append(novel_feature)
                    novel_label_list.append(labels)
                    output_list.append(novel_feature)
                    label_list.append(labels)
                output_list = torch.cat(output_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                # prototype
                label_list = label_list - label_list.min()
                label_count = torch.bincount(label_list)
                num_class = label_list.max() - label_list.min() + 1
                one_hot_label = F.one_hot(label_list.long(), num_classes=num_class).float()
                mean = torch.matmul(one_hot_label.T, output_list)
                mean = mean / label_count[:, None]
                # cov
                cov_list = []
                for class_index in range(num_class):
                    data_index = (label_list == class_index).nonzero()
                    features_this = output_list[data_index.squeeze(-1)]
                    if features_this.size(0) > 1:
                        cov_this = torch.cov(features_this.T)
                        cov_this = torch.std(features_this, dim=0, unbiased=False)
                    else:
                        cov_this = torch.zeros(features_this.size(1)).cuda()
                    cov_list.append(cov_this)
                cov = torch.stack(cov_list, dim=0)
                novel_cov.append(cov)
                novel_mean.append(mean)
            novel_feature_list = torch.cat(novel_feature_list, dim=0)
            novel_label_list = torch.cat(novel_label_list, dim=0)
            final_cov = torch.mean(torch.stack(novel_cov), dim=0)
            final_mean = torch.mean(torch.stack(novel_mean), dim=0)
            return final_mean,final_cov,novel_feature_list,novel_label_list

    def calibration(self,novel_feature_mean,novel_feature_cov,labels):
        base_feature_mean = self.base_feature_mean
        base_feature_cov = self.base_feature_cov
        K=self.args.base_class
        a=self.args.calibration_alpha
        b=1-a
        c=self.args.calibration_gamma
        t=self.args.calibration_t

        if self.args.is_MPC == True:
            # prototype
            labels = torch.tensor(labels, dtype=torch.long)
            with torch.no_grad():
                output_novel_feature_mean = self.ProtoComNet(novel_feature_mean,labels,self.attribute_feature_mean,phase="test")
            novel_feature_mean = a * novel_feature_mean + b * output_novel_feature_mean
        # cov
        if self.args.is_AUG == True:
            cost = cosine_similarity(novel_feature_mean.cpu(), base_feature_mean.cpu())
            cost = torch.tensor(cost).cuda()
            topk_cost, topk_indices = torch.topk(cost, K, dim=1)
            topk_weight = F.softmax(t * topk_cost, dim=1)

            fusion_cov_matrices = []
            for i in range(topk_weight.shape[0]):
                w = topk_weight[i, :]
                weighted_cov = torch.matmul(w, base_feature_cov[topk_indices[i]])
                fusion_cov_matrices.append(weighted_cov)
            novel_feature_cov = c * (novel_feature_cov + torch.stack(fusion_cov_matrices, dim=0))

        return novel_feature_mean,novel_feature_cov

    def align_loss(self, logits, label_rep, novel_class_start):
        novel_classes_idx = torch.argwhere(label_rep >= novel_class_start).flatten()
        base_classes_idx = torch.argwhere(label_rep < novel_class_start).flatten()
        novel_loss = base_loss = 0
        criterion = nn.CrossEntropyLoss()
        if novel_classes_idx.numel() != 0:
            novel_loss = criterion(logits[novel_classes_idx, :], label_rep[novel_classes_idx])
        if base_classes_idx.numel() != 0:
            base_loss = criterion(logits[base_classes_idx, :], label_rep[base_classes_idx])
        loss = (novel_loss * self.args.novel_weight) + (base_loss * self.args.base_weight)
        return loss

    def contrastive_loss(self,projs,labels,session):
        sc_criterion = supcon.SupConLoss()
        # Anchor point
        targets = self.prototype_classifier.get_classifier_weights().detach().clone()
        targets_label = torch.arange(targets.shape[0]).cuda()
        if session==0:
            logits = torch.cat((projs, targets), dim=0)
            labels = torch.cat((labels, targets_label), dim=0)
            random_indices = torch.randperm(logits.size(0))
            logits = logits[random_indices]
            labels = labels[random_indices]
            logits = logits.unsqueeze(1)

        else:
            if self.args.contrastive_loss_mode=="only_novel":
                logits = torch.cat((projs,targets[self.args.base_class:self.args.base_class + session * self.args.way,:]), dim=0)
                labels = torch.cat((labels,targets_label[self.args.base_class:self.args.base_class + session * self.args.way]), dim=0)
                random_indices = torch.randperm(logits.size(0))
                logits = logits[random_indices]
                labels = labels[random_indices]
                logits = logits.unsqueeze(1)
            else:
                logits = torch.cat((projs, targets),dim=0)
                labels = torch.cat((labels, targets_label), dim=0)
                random_indices = torch.randperm(logits.size(0))
                logits = logits[random_indices]
                labels = labels[random_indices]
                logits = logits.unsqueeze(1)

        loss = sc_criterion(logits, labels)
        return loss

    def contrastive_loss_pretrain(self,logits,labels):
        sc_criterion = supcon.SupConLoss(temperature=0.07)
        logits = logits.unsqueeze(1)
        loss = sc_criterion(logits, labels)
        return loss