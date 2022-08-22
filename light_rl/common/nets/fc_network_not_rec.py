import torch

from light_rl.common.nets.fc_network import FCNetwork


class FCNetworkNotRec(FCNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward((x, None))[0]