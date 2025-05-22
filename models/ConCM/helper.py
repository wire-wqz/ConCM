from torch.utils.data import Dataset
import math
import torch.nn.functional as F
import torch

def get_eft_matrix(prototypes):
    prototypes=prototypes.cpu()
    num_classes = prototypes.shape[0]

    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    M = i_nc_nc - one_nc_nc
    U_svd, _, Vt = torch.svd(prototypes.T@M)
    orth_vec = U_svd @ Vt.T
    orth_vec=F.normalize(orth_vec , p=2, dim=0)

    etf_vec = torch.mul(torch.matmul(orth_vec, M),math.sqrt(num_classes / (num_classes - 1)))
    etf_vec=etf_vec.cuda()
    etf_vec=etf_vec.T
    return etf_vec


class GaussianDataset_cov(Dataset):
    def __init__(self, means, covs, sample_num_base,sample_num_novel,label_base,novel_feas,novel_labels):
        self.samples = novel_feas
        self.labels = novel_labels
        for label, (mean, cov) in enumerate(zip(means, covs)):
            if label<label_base:
                samples_num=sample_num_base
            else:
                samples_num=sample_num_novel
            epsilon = 1e-5
            cov = cov + epsilon * torch.eye(cov.shape[0], device=cov.device)
            mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            class_samples = mvn.sample((samples_num,))
            self.samples = torch.cat((self.samples,class_samples), dim=0)
            self.labels = torch.cat((self.labels,torch.full((samples_num,), label).cuda()), dim=0)
            self.labels = self.labels.long()

        self.samples=self.samples.cpu().numpy()
        self.labels = self.labels.cpu().numpy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
