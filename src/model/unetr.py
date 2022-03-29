from typing import Union, Sequence
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
import torch
from torch import optim
import pytorch_lightning as pl
from monai.data import decollate_batch

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyUNETR(pl.LightningModule):
    def __init__(self, 
                img_size:Union[int, Sequence[int]] = (96, 96, 96),
                feature_size: int = 16,
                hidden_size: int = 768,
                mlp_dim: int = 3072,
                num_heads: int = 12,
                dropout_rate: float = 0.0,
                lr:float = 1e-4
        ) -> None:

        super(MyUNETR, self).__init__()

        self._model = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=dropout_rate,
        )
 
        self.lr = lr
        self.roi_size = img_size

        self.loss_function = DiceCELoss(sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        self.post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.metric_values = []
        self.epoch_loss_values = []
    
    def summary(self) -> str:
        return self._model.__repr__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-7,
                verbose=True,
            ),
            "monitor": "val_epoch_loss",
        }
        return [self.optimizer], self.scheduler
        # return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["label"])
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("train_epoch_loss", avg_loss.to("cpu").detach().numpy().item(), logger=True)
        self.epoch_loss_values.append(avg_loss.to("cpu").detach().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.roi_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        self.log("val_epoch_loss", mean_val_loss, logger=True)
        self.log("val_epoch_dice", mean_val_dice, logger=True)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"\ncurrent epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)

