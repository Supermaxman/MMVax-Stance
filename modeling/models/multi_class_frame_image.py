import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, BridgeTowerModel, CLIPModel, FlavaModel, ViltModel

from modeling.metrics import Metric
from modeling.thresholds import ThresholdModule


class MultiClassFrameImageModel(pl.LightningModule, ABC):
    def __init__(
        self,
        pre_model_name: str,
        label_map: Dict[str, int],
        threshold: ThresholdModule,
        metric: Metric,
        num_threshold_steps: int = 100,
        update_threshold: bool = False,
        dropout_prob: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        lr_warm_up: float = 0.1,
        load_pre_model: bool = True,
        torch_cache_dir: str = None,
    ):
        super().__init__()
        self.pre_model_name = pre_model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_warm_up = lr_warm_up
        self.torch_cache_dir = torch_cache_dir
        self.load_pre_model = load_pre_model
        self.outputs = []

        self.label_map = label_map
        self.num_classes = len(label_map)
        self.threshold = threshold
        self.num_threshold_steps = num_threshold_steps
        self.update_threshold = update_threshold
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.f_dropout = torch.nn.Dropout(p=dropout_prob)
        self.metric = metric

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.score_func = torch.nn.Softmax(dim=-1)

    @abstractmethod
    def forward(self, batch):
        pass

    def eval_epoch_end(self, outputs, stage):
        loss = torch.cat([x["loss"] for x in outputs], dim=0).mean().cpu()
        self.log(f"{stage}_loss", loss)
        self.threshold.cpu()

        results, labels, preds, t_ids = self.eval_outputs(
            outputs, stage, self.num_threshold_steps, self.update_threshold
        )
        for val_name, val in results.items():
            self.log(val_name, val)

        self.threshold.to(self.device)

    def eval_outputs(
        self, outputs, stage, num_threshold_steps=100, update_threshold=True
    ):
        results = {}

        t_ids = self.flatten([x["ids"] for x in outputs])
        # [count]
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()
        # [count, num_classes]
        scores = torch.cat([x["scores"] for x in outputs], dim=0).cpu()

        self.threshold.cpu()
        if update_threshold:
            m_min_score = torch.min(scores).item()
            m_max_score = torch.max(scores).item()
            # check 100 values between min and max
            if abs(m_min_score - m_max_score) < 1e-6:
                m_max_score += 10
            m_delta = (m_max_score - m_min_score) / num_threshold_steps
            max_threshold, max_metrics = self.metric.best(
                labels,
                scores,
                self.threshold,
                threshold_min=m_min_score,
                threshold_max=m_max_score,
                threshold_delta=m_delta,
            )
            self.threshold.update_thresholds(max_threshold)
        preds = self.threshold(scores)

        f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(labels, preds)

        results[f"{stage}_f1"] = f1
        results[f"{stage}_p"] = p
        results[f"{stage}_r"] = r

        for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
            label_name = self.inv_label_map[cls_index]
            results[f"{stage}_{label_name}_f1"] = c_f1
            results[f"{stage}_{label_name}_p"] = c_p
            results[f"{stage}_{label_name}_r"] = c_r
            results[f"{stage}_{cls_index}_f1"] = c_f1
            results[f"{stage}_{cls_index}_p"] = c_p
            results[f"{stage}_{cls_index}_r"] = c_r

        return results, labels, preds, t_ids

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        logits = self(batch)
        loss = self.loss(logits, batch["labels"])
        scores = self.score_func(logits)
        results = {
            # [bsize]
            "ids": batch["ids"],
            "labels": batch["labels"],
            "logits": logits,
            "loss": loss,
            "scores": scores,
        }
        return results

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        batch_logits = self(batch)
        batch_labels = batch["labels"]
        batch_loss = self.loss(batch_logits, batch_labels)
        loss = batch_loss.mean()
        self.log("train_loss", loss)
        result = {"loss": loss}
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        results = self.eval_step(batch, batch_idx, dataloader_idx)
        return results

    @staticmethod
    def flatten(multi_list):
        return [item for sub_list in multi_list for item in sub_list]

    def configure_optimizers(self):
        params = self._get_optimizer_params(self.weight_decay)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = WarmupLR(
            optimizer,
            num_warmup_steps=int(
                math.ceil(self.lr_warm_up * self.trainer.estimated_stepping_batches)
            ),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        optimizer_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
            },
        }

        return optimizer_dict

    def _get_optimizer_params(self, weight_decay):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_params

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "fit":
            self.update_threshold = True

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch_outputs = self.eval_step(batch, batch_idx, dataloader_idx)
        self.outputs.append(batch_outputs)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        batch_outputs = self.eval_step(batch, batch_idx, dataloader_idx)
        self.outputs.append(batch_outputs)

    def on_validation_epoch_end(self):
        self.eval_epoch_end(self.outputs, "val")
        self.outputs.clear()

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.outputs, "test")
        self.outputs.clear()


class MultiClassFrameImageBridgeTowerModel(MultiClassFrameImageModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.load_pre_model:
            self.model = BridgeTowerModel.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
            self.model = BridgeTowerModel.from_config(config)

        # noinspection PyUnresolvedReferences
        self.hidden_size = self.model.config.hidden_size

        # 2 * hidden_size because we are concatenating the image and text pooled states
        self.cls_layer = torch.nn.Linear(
            in_features=2 * self.hidden_size, out_features=self.num_classes
        )

    def forward(self, batch):
        model_batch = {k: v for k, v in batch.items() if k not in {"ids", "labels"}}

        # 'text_features' -> [bsize, seq_len, hidden_size]
        # 'image_features' -> [bsize, img_seq_len, hidden_size]
        # 'pooler_output' -> [bsize, 2 * hidden_size]
        outputs = self.model(**model_batch)
        pooled_output = outputs["pooler_output"]
        pooled_output = self.f_dropout(pooled_output)
        # [bsize, num_classes]
        logits = self.cls_layer(pooled_output)
        return logits


class MultiClassFrameImageClipJointModel(MultiClassFrameImageModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.load_pre_model:
            self.model = CLIPModel.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
            self.model = CLIPModel.from_config(config)

        self.hidden_size = self.model.config.projection_dim

        self.cls_layer = torch.nn.Linear(
            in_features=2 * self.hidden_size, out_features=self.num_classes
        )

    def forward(self, batch):
        model_batch = {k: v for k, v in batch.items() if k not in {"ids", "labels"}}

        # 'text_embeds' -> [bsize, hidden_size]
        # 'image_embeds' -> [bsize, hidden_size]
        outputs = self.model(**model_batch)
        text_embeds = outputs["text_embeds"]
        image_embeds = outputs["image_embeds"]
        pooled_output = torch.cat([text_embeds, image_embeds], dim=-1)
        pooled_output = self.f_dropout(pooled_output)
        # [bsize, num_classes]
        logits = self.cls_layer(pooled_output)
        return logits


class MultiClassFrameImageClipTextModel(MultiClassFrameImageModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.load_pre_model:
            self.model = CLIPModel.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
            self.model = CLIPModel.from_config(config)

        self.hidden_size = self.model.config.projection_dim

        self.cls_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_classes
        )

    def forward(self, batch):
        model_batch = {k: v for k, v in batch.items() if k not in {"ids", "labels"}}

        # 'text_embeds' -> [bsize, hidden_size]
        # 'image_embeds' -> [bsize, hidden_size]
        outputs = self.model(**model_batch)
        pooled_output = outputs["text_embeds"]
        pooled_output = self.f_dropout(pooled_output)
        # [bsize, num_classes]
        logits = self.cls_layer(pooled_output)
        return logits


class MultiClassFrameImageClipImageModel(MultiClassFrameImageModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.load_pre_model:
            self.model = CLIPModel.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
            self.model = CLIPModel.from_config(config)

        self.hidden_size = self.model.config.projection_dim

        self.cls_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_classes
        )

    def forward(self, batch):
        model_batch = {k: v for k, v in batch.items() if k not in {"ids", "labels"}}

        # 'text_embeds' -> [bsize, hidden_size]
        # 'image_embeds' -> [bsize, hidden_size]
        outputs = self.model(**model_batch)
        pooled_output = outputs["image_embeds"]
        pooled_output = self.f_dropout(pooled_output)
        # [bsize, num_classes]
        logits = self.cls_layer(pooled_output)
        return logits


class MultiClassFrameImageViltModel(MultiClassFrameImageModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.load_pre_model:
            self.model = ViltModel.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
            self.model = ViltModel.from_config(config)

        self.hidden_size = self.model.config.hidden_size
        self.cls_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_classes
        )

    def forward(self, batch):
        model_batch = {k: v for k, v in batch.items() if k not in {"ids", "labels"}}

        # 'text_embeds' -> [bsize, hidden_size]
        # 'image_embeds' -> [bsize, hidden_size]
        outputs = self.model(**model_batch)
        pooled_output = outputs["pooler_output"]
        pooled_output = self.f_dropout(pooled_output)
        # [bsize, num_classes]
        logits = self.cls_layer(pooled_output)
        return logits


class MultiClassFrameImageFlavaModel(MultiClassFrameImageModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.load_pre_model:
            self.model = FlavaModel.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pre_model_name, cache_dir=self.torch_cache_dir
            )
            self.model = FlavaModel.from_config(config)

        self.hidden_size = self.model.config.hidden_size
        self.cls_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_classes
        )

    def forward(self, batch):
        model_batch = {k: v for k, v in batch.items() if k not in {"ids", "labels"}}

        # 'image_embeddings' -> [bsize, img_seq_len, hidden_size]
        # 'text_embeddings' -> [bsize, seq_len, hidden_size]
        # 'multimodal_embeddings' -> [bsize, seq_len + img_seq_len, hidden_size]
        outputs = self.model(**model_batch)
        # cls embedding is first token
        pooled_output = outputs["multimodal_embeddings"][:, 0, :]
        pooled_output = self.f_dropout(pooled_output)
        # [bsize, num_classes]
        logits = self.cls_layer(pooled_output)
        return logits


class WarmupLR(LambdaLR):
    def __init__(
        self, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch=-1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(
            optimizer=optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )
