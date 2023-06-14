import math
from abc import ABC, abstractmethod
from typing import Callable, Type, Union

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    BertForPreTraining,
)


class BasePreModel(pl.LightningModule, ABC):
    lm: Callable

    def __init__(
        self,
        pre_model_name: str,
        pre_model_type: Type[
            Union[
                AutoModel,
                AutoModelForSequenceClassification,
                AutoModelForSeq2SeqLM,
                AutoModelForMaskedLM,
                BertForPreTraining,
            ]
        ] = AutoModel,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        lr_warm_up: float = 0.1,
        load_pre_model: bool = True,
        torch_cache_dir: str = None,
    ):
        r"""
        Base class for Pre-Trained Models.

        Args:

                pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

                pre_model_type: Type of pre-trained model.
                        Default: [`AutoModel`].

                learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
                        ``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly
                        over the remaining
                        ``1.0-lr_warm_up`` training steps.

                weight_decay: How much weight decay to apply in the AdamW optimizer.
                        Default: ``0.0``.

                lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
                        Default: ``0.1``.

                load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be
                        initialized.
                        Cuts down on model load time if you plan on loading your model from a checkpoint, as there is
                        no reason to
                        initialize your model twice.
                        Default: ``True``.

                torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
                        Default: ``None``.

        """

        super().__init__()
        self.pre_model_name = pre_model_name
        self.pre_model_type = pre_model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_warm_up = lr_warm_up
        # assigned later when training starts
        self.torch_cache_dir = torch_cache_dir
        if load_pre_model:
            self.lm = self.pre_model_type.from_pretrained(
                pre_model_name, cache_dir=torch_cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                pre_model_name, cache_dir=torch_cache_dir
            )
            self.lm = self.pre_model_type.from_config(config)
        self.outputs = []

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

    @abstractmethod
    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    @abstractmethod
    def eval_epoch_end(self, outputs, stage):
        pass

    @abstractmethod
    def lm_step(self, input_ids, attention_mask, token_type_ids=None):
        pass


class BaseLanguageModel(BasePreModel, ABC):
    def __init__(
        self,
        pre_model_name: str,
        pre_model_type: Type[
            Union[AutoModel, AutoModelForSequenceClassification]
        ] = AutoModel,
        *args,
        **kwargs
    ):
        r"""
        Base class for Pre-Trained Language Models.

        Args:

                pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

                pre_model_type: Type of pre-trained model.
                        Default: [`AutoModel`].

                learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
                        ``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly
                        over the remaining
                        ``1.0-lr_warm_up`` training steps.

                weight_decay: How much weight decay to apply in the AdamW optimizer.
                        Default: ``0.0``.

                lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
                        Default: ``0.1``.

                load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be
                        initialized.
                        Cuts down on model load time if you plan on loading your model from a checkpoint, as there is
                        no reason to
                        initialize your model twice.
                        Default: ``True``.

                torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
                        Default: ``None``.

        """

        super().__init__(pre_model_name, pre_model_type, *args, **kwargs)

        # TODO check for these, not all models may have them
        # noinspection PyUnresolvedReferences
        self.hidden_size = self.lm.config.hidden_size
        # noinspection PyUnresolvedReferences
        self.hidden_dropout_prob = self.lm.config.hidden_dropout_prob

    def lm_step(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        contextualized_embeddings = outputs[0]
        return contextualized_embeddings


class BaseLanguageModelForSequenceClassification(BasePreModel, ABC):
    def __init__(
        self,
        pre_model_name: str,
        pre_model_type: Type[
            Union[AutoModel, AutoModelForSequenceClassification]
        ] = AutoModelForSequenceClassification,
        *args,
        **kwargs
    ):
        r"""
        Base class for Pre-Trained Language Models for Sequence Classification.

        Args:

                pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

                pre_model_type: Type of pre-trained model.
                        Default: [`AutoModel`].

                learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
                        ``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly
                        over the remaining
                        ``1.0-lr_warm_up`` training steps.

                weight_decay: How much weight decay to apply in the AdamW optimizer.
                        Default: ``0.0``.

                lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
                        Default: ``0.1``.

                load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be
                        initialized.
                        Cuts down on model load time if you plan on loading your model from a checkpoint, as there is
                        no reason to
                        initialize your model twice.
                        Default: ``True``.

                torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
                        Default: ``None``.

        """

        super().__init__(pre_model_name, pre_model_type, *args, **kwargs)
        # TODO check for these, not all models may have them
        # noinspection PyUnresolvedReferences
        # self.id2label = self.lm.config.id2label
        # noinspection PyUnresolvedReferences
        # self.label2id = self.lm.config.label2id
        # 0 - contradiction
        # 1 - neutral
        # 2 - entailment
        # want
        # 0 - neutral
        # 1 - entailment
        # 2 - contradiction
        # map
        # 1 -> 0
        # 2 -> 1
        # 0 -> 2
        # TODO build automatically
        # self.label_list = [1, 2, 0]

    def lm_step(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs[0]
        # re-arrange logits
        # logits = logits[:, self.label_list]
        return logits


class BaseLanguageModelForSeq2SeqLM(BasePreModel, ABC):
    def __init__(
        self,
        pre_model_name: str,
        pre_model_type: Type[AutoModelForSeq2SeqLM] = AutoModelForSeq2SeqLM,
        *args,
        **kwargs
    ):
        r"""
        Base class for Pre-Trained Language Models for sequence 2 sequence.

        Args:

                pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

                pre_model_type: Type of pre-trained model.
                        Default: [`AutoModelForSeq2SeqLM`].

                learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
                        ``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly
                        over the remaining
                        ``1.0-lr_warm_up`` training steps.

                weight_decay: How much weight decay to apply in the AdamW optimizer.
                        Default: ``0.0``.

                lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
                        Default: ``0.1``.

                load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be
                initialized.
                        Cuts down on model load time if you plan on loading your model from a checkpoint, as there is
                        no reason to
                        initialize your model twice.
                        Default: ``True``.

                torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
                        Default: ``None``.

        """

        super().__init__(pre_model_name, pre_model_type, *args, **kwargs)

    def lm_step(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        contextualized_embeddings = outputs[0]
        return contextualized_embeddings


class BaseLanguageModelForPreTraining(BasePreModel, ABC):
    def __init__(
        self,
        pre_model_name: str,
        pre_model_type: Type[BertForPreTraining] = BertForPreTraining,
        *args,
        **kwargs
    ):
        r"""
        Base class for Masked Language Modeling.

        Args:

                pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

                pre_model_type: Type of pre-trained model.
                        Default: [`BertForPreTraining`].

                learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
                        ``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly
                         over the remaining
                        ``1.0-lr_warm_up`` training steps.

                weight_decay: How much weight decay to apply in the AdamW optimizer.
                        Default: ``0.0``.

                lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
                        Default: ``0.1``.

                load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be
                        initialized.
                        Cuts down on model load time if you plan on loading your model from a checkpoint, as there is
                        no reason to
                        initialize your model twice.
                        Default: ``True``.

                torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
                        Default: ``None``.

        """

        super().__init__(pre_model_name, pre_model_type, *args, **kwargs)

    def lm_step(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        contextualized_embeddings = outputs[0]
        return contextualized_embeddings

    def configure_optimizers(self):
        params = self._get_optimizer_params(self.weight_decay)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer


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
