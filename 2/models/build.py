import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F
from util.scheduler import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler import CosineLRScheduler, PolyLRScheduler
import math
import torchmetrics
from util.metrics import *
from tabulate import tabulate
from pathlib import Path
import timm
import cv2

import torch.linalg as LA


class seg_network(pl.LightningModule):

    def __init__(self, model, opt, cfg):
        super().__init__()
        self.model = model
        self.opt = opt
        self.cfg = cfg

        if len(cfg["MODEL"]["load_Pretraining"].strip()) != 0:
            print("loading the model from %s" % cfg["MODEL"]["load_Pretraining"])
            state_dict = torch.load(
                cfg["MODEL"]["load_Pretraining"], map_location="cpu"
            )
            self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch, batch_idx):

        x = batch
        _, H, W = x["label"].shape
        out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))

        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        loss = F.cross_entropy(
            out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
        )

        self.log(
            "Loss/train_segloss",
            loss,
            on_step=True,
            sync_dist=True,
            batch_size=self.cfg["TRAIN"]["batch_size"],
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_start(self):
        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:

            self.CVC_300_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ClinicDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ColonDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.ETIS_LaribPolypDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.Kvasir_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

        else:
            self.metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

        self.eval_last_path = os.path.join(
            self.opt.save_pth_dir,
            "eval_last_{}.txt".format(self.cfg["DATASET"]["dataset"]),
        )
        with open(self.eval_last_path, "a") as f:
            f.write(
                "\n\n\n!!!!!! Starting validation for epoch {} !!!!!\n".format(
                    self.current_epoch
                )
            )

    def validation_step(self, batch, batch_idx):
        x = batch

        _, H, W = x["label"].shape
        out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))

        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        loss = F.cross_entropy(
            out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
        )

        self.log(
            "Loss/validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

        out = out.softmax(dim=1)

        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:

            if "CVC-300" in x["img_path"][0]:
                self.CVC_300_metrics.update(out, x["label"])

            elif "CVC-ClinicDB" in x["img_path"][0]:
                self.CVC_ClinicDB_metrics.update(out, x["label"])

            elif "CVC-ColonDB" in x["img_path"][0]:
                self.CVC_ColonDB_metrics.update(out, x["label"])

            elif "ETIS-LaribPolypDB" in x["img_path"][0]:
                self.ETIS_LaribPolypDB_metrics.update(out, x["label"])

            elif "Kvasir" in x["img_path"][0]:
                self.Kvasir_metrics.update(out, x["label"])
        else:
            self.metrics.update(out, x["label"])
        return loss

    def on_validation_epoch_end(self):

        average_IoU = 0.0

        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
            sem_index = self.CVC_300_metrics.compute()
            self.log(
                "index/CVC_300_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_300_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_300_F1score", sem_index["mF1"], sync_dist=True, batch_size=1
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} CVC_300 images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.CVC_ClinicDB_metrics.compute()
            self.log(
                "index/CVC_ClinicDB_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ClinicDB_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ClinicDB_F1score",
                sem_index["mF1"],
                sync_dist=True,
                batch_size=1,
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} CVC_ClinicDB images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.CVC_ColonDB_metrics.compute()
            self.log(
                "index/CVC_ColonDB_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ColonDB_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ColonDB_F1score",
                sem_index["mF1"],
                sync_dist=True,
                batch_size=1,
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} CVC_ColonDB images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.ETIS_LaribPolypDB_metrics.compute()
            self.log(
                "index/ETIS_LaribPolypDB_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/ETIS_LaribPolypDB_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/ETIS_LaribPolypDB_F1score",
                sem_index["mF1"],
                sync_dist=True,
                batch_size=1,
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} ETIS_LaribPolypDB images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.Kvasir_metrics.compute()
            self.log(
                "index/Kvasir_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/Kvasir_Accuracy", sem_index["mACC"], sync_dist=True, batch_size=1
            )
            self.log(
                "index/Kvasir_F1score", sem_index["mF1"], sync_dist=True, batch_size=1
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} Kvasir images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            average_IoU = average_IoU / 5

            self.log(
                "index/average_IoU",
                average_IoU,
                sync_dist=True,
                batch_size=1,
                prog_bar=True,
            )

        else:
            sem_index = self.metrics.compute()
            self.log(
                "index/average_IoU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
                prog_bar=True,
            )
            self.log("index/Accuracy", sem_index["mACC"], sync_dist=True, batch_size=1)
            self.log("index/F1score", sem_index["mF1"], sync_dist=True, batch_size=1)

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n============== Eval on {} {} images =================\n".format(
                        self.cfg["MODEL"]["model_names"], self.cfg["DATASET"]["dataset"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

    def on_test_epoch_start(self):
        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:

            self.CVC_300_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ClinicDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ColonDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.ETIS_LaribPolypDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.Kvasir_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

        else:
            self.metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

    def test_step(self, batch, batch_idx):
        x = batch

        _, H, W = x["label"].shape
        out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))

        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        loss = F.cross_entropy(
            out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
        )

        out = out.softmax(dim=1)
        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:

            if "CVC-300" in x["img_path"][0]:
                self.CVC_300_metrics.update(out, x["label"])

            elif "CVC-ClinicDB" in x["img_path"][0]:
                self.CVC_ClinicDB_metrics.update(out, x["label"])

            elif "CVC-ColonDB" in x["img_path"][0]:
                self.CVC_ColonDB_metrics.update(out, x["label"])

            elif "ETIS-LaribPolypDB" in x["img_path"][0]:
                self.ETIS_LaribPolypDB_metrics.update(out, x["label"])

            elif "Kvasir" in x["img_path"][0]:
                self.Kvasir_metrics.update(out, x["label"])
        else:
            self.metrics.update(out, x["label"])
        return loss

    def on_test_epoch_end(self):
        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
            sem_index = self.CVC_300_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== CVC_300 =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.CVC_ClinicDB_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== CVC_ClinicDB =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.CVC_ColonDB_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== CVC_ColonDB =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.ETIS_LaribPolypDB_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== ETIS_LaribPolypDB =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.Kvasir_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== Kvasir =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

        else:
            sem_index = self.metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["OPTIMIZER"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.cfg["OPTIMIZER"]["weight_decay"],
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            self.cfg["TRAIN"]["nepoch"],
            self.cfg["SCHEDULER"]["warmup_epoch"],
            math.ceil(self.opt.total_samples / int(self.cfg["TRAIN"]["node"])),
            self.cfg["SCHEDULER"]["lr_warmup"],
            self.cfg["SCHEDULER"]["warmup_ratio"],
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler_config)
