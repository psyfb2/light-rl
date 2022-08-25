import torch

from torch import nn, Tensor

from light_rl.common.nets.fc_network import FCNetwork
from light_rl.common.nets.virtual_batch_norm import VirtualBatchNorm1d


class FCNetworkVBN(FCNetwork):
    def __init__(self, ref_x: torch.Tensor, *args, **kwargs):
        """ Creates a FCNetwork, but puts 
        Virtual Batch Normalization before any ReLU layer.

        Args:
            ref_x (torch.Tensor): reference batch to use for VBN
            *args, **kwargs to FCNetwork
        """
        super().__init__(*args, **kwargs)
        self.ref_x = ref_x
        # insert VBN before reach relu layer
        i = 0
        while i < len(self._layers):
            if isinstance(self._layers[i], nn.ReLU):
                self._layers.insert(i, VirtualBatchNorm1d(self._layers[i - 1].out_features))
                i += 1
            i += 1
    
    def forward(self, x):
        x, hx_cx_lst = x

        added_batch_dim = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            added_batch_dim = True

        if self._lstm_count == 0:
            hx_cx_lst = []

        zeros = lambda x : torch.zeros(self.lstm_hidden_size
                ) if x.ndim == 1 else torch.zeros(x.shape[0], self.lstm_hidden_size)
        if hx_cx_lst is None:
            # initialise hx, cx to be zeros for all lstm's (account for batch size)
            hx_cx_lst = [(zeros(x), zeros(x)) for _ in range(self._lstm_count)]

        if len(hx_cx_lst) != self._lstm_count:
            raise ValueError(
                f"recurrent state list {hx_cx_lst} must have length {self._lstm_count} "
                f"not length {len(hx_cx_lst)}, as this is the number of lstm's in this network"
            )
        
        # reference pass (VBN needs two forward passes)
        ref_means = []
        lstm_counter = 0
        ref_x = self.ref_x
        for layer in self._layers:
            if isinstance(layer, nn.LSTMCell):
                prev_hx, prev_cx = hx_cx_lst[lstm_counter]
                state = (prev_hx.tile((ref_x.size(0), 1)), prev_cx.tile((ref_x.size(0), 1)))
                hx, cx = layer(ref_x, state)
                ref_x = hx
                lstm_counter += 1
            elif isinstance(layer, VirtualBatchNorm1d):
                ref_x, mean, meansq = layer(ref_x, None, None)
                ref_means.append((mean, meansq))
            else:
                ref_x = layer(ref_x)
        
        # train pass
        ref_means_counter = 0
        lstm_counter = 0
        new_hx_cx_lst = []
        for layer in self._layers:
            if isinstance(layer, nn.LSTMCell):
                hx, cx = layer(x, hx_cx_lst[lstm_counter])
                new_hx_cx_lst.append((hx, cx))
                x = hx
                lstm_counter += 1
            elif isinstance(layer, VirtualBatchNorm1d):
                x, _, _ = layer(x, *ref_means[ref_means_counter])
                ref_means_counter += 1
            else:
                x = layer(x)
        
        if added_batch_dim:
            x = x.squeeze(0)

        return x, new_hx_cx_lst



    