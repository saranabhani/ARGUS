import math
import torch
from torch import nn

    
class TemperatureScaler(nn.Module):
    def __init__(self, init_T=1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float32))

    def forward(self, logits):
        return logits / self.log_T.exp()

    @property
    def temperature(self) -> float:
        return float(self.log_T.exp().detach().cpu().item())

def nll_vs_soft(logits, soft_targets):
    logp = torch.log_softmax(logits, dim=-1)
    return (-soft_targets * logp).sum(dim=-1).mean()

def nll_ce_from_logits(logits, y_true):
    return nn.functional.cross_entropy(logits, y_true)

def temperature_scaling(logits, labels, training_type='soft'):
    temp = TemperatureScaler(init_T=1.0).to(logits.device)
    opt = torch.optim.LBFGS(temp.parameters(), lr=0.01, max_iter=50, line_search_fn="strong_wolfe")
    if training_type == 'soft':
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nll_vs_soft(temp(logits), labels)
            loss.backward()
            return loss
    else:
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nll_ce_from_logits(temp(logits), labels)
            loss.backward()
            return loss
    opt.step(closure)
    T_value = temp.temperature
    print(f"[Calibration] Learned temperature T = {T_value:.4f}")
    return T_value, temp
