from transformers import Trainer
from torch import nn
import torch

class SoftLabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        if class_weights is not None:
            self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        targets = inputs.pop("labels")
        _ = inputs.pop("hard_label", None)

        outputs = model(**inputs)
        logits = outputs.logits                           
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        targets = targets.to(dtype=log_probs.dtype, device=log_probs.device).clamp_min(1e-8)

        if self.class_weights is not None:
            w = self.class_weights.to(device=log_probs.device, dtype=log_probs.dtype)
            weighted_targets = targets * w
            weighted_targets = weighted_targets / weighted_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            weighted_targets = targets

        loss = self.loss_fn(log_probs, weighted_targets)

        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        return loss