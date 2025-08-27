import torch

import torchmetrics


class Metrics(torchmetrics.Metric):
    def __init__(self, num_classes: int, ignore_label: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.add_state(
            "hist", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum"
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label

        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)

    def compute(self):
        ious = self.hist.diag() / (
            self.hist.sum(0) + self.hist.sum(1) - self.hist.diag()
        )
        ious[ious.isnan()] = 0.0
        miou = ious.mean().item()

        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()] = 0.0
        mf1 = f1.mean().item()

        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()] = 0.0
        macc = acc.mean().item()

        return {
            "IOUs": ious.cpu().numpy().round(4).tolist(),
            "mIOU": round(miou, 4),
            "F1": f1.cpu().numpy().round(2).tolist(),
            "mF1": round(mf1, 4),
            "ACC": acc.cpu().numpy().round(4).tolist(),
            "mACC": round(macc, 4),
        }
