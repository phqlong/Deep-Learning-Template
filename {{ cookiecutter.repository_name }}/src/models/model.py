from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import Accuracy, F1Score, Precision, Recall


class MNISTLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Save net model to caculate self.parameters() in configure_optimizers
        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_f1 = F1Score(task="multiclass", average="weighted", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", average="weighted", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", average="weighted", num_classes=num_classes)

        self.test_f1 = F1Score(task="multiclass", average="weighted", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", average="weighted", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", average="weighted", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from 
        # all batches of the epoch this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        val_loss = self.val_loss(loss)
        val_acc = self.val_acc(preds, targets)
        val_f1 = self.val_f1(preds, targets)
        val_precision = self.val_precision(preds, targets)
        val_recall = self.val_recall(preds, targets)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", val_recall, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        epoch_acc = self.val_acc.compute()  # get current val acc, last acc of epoch
        self.val_acc_best(epoch_acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        test_loss = self.test_loss(loss)
        test_acc = self.test_acc(preds, targets)
        test_f1 = self.test_f1(preds, targets)
        test_precision = self.test_precision(preds, targets)
        test_recall = self.test_recall(preds, targets)
        self.log("test/loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", test_recall, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Lr and weight_decay are partially initialized in hydra.utils.instantiate(cfg.model)
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "default.yaml")
    _ = hydra.utils.instantiate(cfg)
