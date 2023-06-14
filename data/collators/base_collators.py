from abc import ABC, abstractmethod

import torch


class BatchCollator(ABC):
    def __init__(self, max_seq_len: int = 512, use_tpus=False):
        self.max_seq_len = max_seq_len
        self.use_tpus = use_tpus

    def _calculate_seq_padding(self, examples, key="input_ids", max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        if self.use_tpus:
            pad_seq_len = max_seq_len
        else:
            pad_seq_len = 0
            for ex in examples:
                if key in ex:
                    pad_seq_len = max(pad_seq_len, min(len(ex[key]), max_seq_len))
        if pad_seq_len == 0:
            pad_seq_len = 1
        return pad_seq_len

    def pad_and_apply(
        self, id_list, id_tensor, ex_idx, max_seq_len=None, dtype=torch.long
    ):
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        ex_ids = id_list[:max_seq_len]
        id_tensor[ex_idx, : len(ex_ids)] = torch.tensor(ex_ids, dtype=dtype)

    @abstractmethod
    def __call__(self, examples: list) -> dict:
        pass
