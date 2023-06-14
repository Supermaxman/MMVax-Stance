import torch
from PIL import Image


class MultiClassFrameImageBatchCollator:
    def __init__(self, processor, max_seq_len: int = 512, use_tpus=False):
        super().__init__()
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.use_tpus = use_tpus

    def __call__(self, examples: list) -> dict:
        batch_size = len(examples)
        # [ex_count, num_classes]
        labels = torch.zeros([batch_size], dtype=torch.long)
        ids = []
        images = []
        texts = []
        for ex_idx, ex in enumerate(examples):
            ids.append(ex["ids"])
            if "label" in ex:
                labels[ex_idx] = ex["label"]
            images.append(Image.open(ex["image_path"]).convert("RGB"))
            texts.append(ex["text"])
        data = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        batch = {
            "ids": ids,
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "pixel_values": data["pixel_values"],
            "labels": labels,
        }
        if "token_type_ids" in data:
            batch["token_type_ids"] = data["token_type_ids"]
        if "pixel_mask" in data:
            batch["pixel_mask"] = data["pixel_mask"]
        return batch
