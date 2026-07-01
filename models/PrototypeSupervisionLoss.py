import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeSupervisionLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, supervised_prototypes, teacher_prototypes):

        supervised_prototypes = F.normalize(supervised_prototypes, dim=1)
        teacher_prototypes = F.normalize(teacher_prototypes, dim=1)


        logits = torch.matmul(supervised_prototypes, teacher_prototypes.T) / self.temperature


        labels = torch.arange(supervised_prototypes.shape[0]).to(logits.device)

        loss = self.loss_fn(logits, labels)
        return loss