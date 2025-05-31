import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, pretrained = True, embedding_dim = 128):
        super(EmbeddingNet, self).__init__()
        backbone  = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

        modules = list(backbone.children())[:-1]  # all layer except the last fc layer

        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 128)

        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.backbone(x)       # shape: [batch_size, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # shape: [batch_size, 2048]

        x = self.fc(x)              # shape [batch_size, embedding_dim]
        x = F.normalize(x, p =2,  dim = 1)   # L2 normalization

        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)

        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_pos = (1-label)*torch.pow(euclidean_distance, 2)
        loss_net = label*torch.pow(torch.clamp(self.margin - euclidean_distance, min = 0.0), 2)

        loss_contrastive = torch.mean(loss_pos + loss_net)

        return loss_contrastive
