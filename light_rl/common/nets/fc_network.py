from multiprocessing.sharedctypes import Value
import torch

from torch import nn, Tensor
from typing import Iterable, Union, Tuple, List


LSTM_STR = 'lstm'

class FCNetwork(nn.Module):
    def __init__(self, dims: Iterable[Union[int, str]], 
                 output_activation: nn.Module = None,
                 lstm_hidden_size: int = 256):
        """ Create MLP with ReLU layers used as hidden activation functions. Can optionally
            have lstm networks by providing string 'lstm'. For example if the 
            dims is [64, 'lstm' 32], get the following net
            64 (input) -> lstm -> 32 (output)

        Args:
            dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers. use 'lstm' to include an LSTM.
            lstm_hidden_size: number of hidden units in LSTM network if one is used.
            output_activation (nn.Module): PyTorch activation function to use after last layer
        """
        super().__init__()
        if dims[0] == LSTM_STR:
            raise ValueError(f"dims[0] cannot be {LSTM_STR}, must be int for input size to network")

        self.input_size = dims[0]
        self.out_size = dims[-1] if dims[-1] != LSTM_STR else lstm_hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self._layers = []

        # example: dims = [64, 128, 'lstm', 32]
        # then layers = [
        #   Linear(in_size=64, out_size=128)
        #   lstm(in_size=128, out_size=lstm_hidden_size), 
        #   Linear(in=lstm_hidden_size, out=32)
        # ]
        get_size = lambda x : lstm_hidden_size if x == LSTM_STR else x
        self._lstm_count = 0
        for i in range(1, len(dims)):
            if dims[i] == LSTM_STR:
                self._lstm_count += 1
                self._layers.append(nn.LSTMCell(get_size(dims[i - 1]), lstm_hidden_size))
            else:
                self._layers.append(nn.Linear(get_size(dims[i - 1]), dims[i]))
                if i != len(dims) - 1: self._layers.append(nn.ReLU())  # not last iter
        
        # add last activation function
        if output_activation: self._layers.append(output_activation())

        self._layers = nn.ModuleList(self._layers)

    def forward(self, x: Union[Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tensor]
            ) -> Union[Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tensor]:
        """ Compute the forward pass through the network

        Args:
            x Union[Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tensor]: 
                if this network does not have any lstm's than x is just a Tensor.
                Otherwise x is a tuple containing input data tensor and 
                list of recurrent state. For example:
                (
                    input_tensor, 
                    [
                        (hidden_lstm_1, context_lstm_1),
                        ...., 
                        (hidden_list_n, context_lstm_n)
                    ]
                )
                list of recurrent state can be None, in which case
                will initialise with zeros. 

        Returns:
            Union[Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tensor]: 
                if this network does not have any lstm's than output is just a Tensor.
                otherwise output is output of forward pass along with 
                any recurrent state. For example:
                (
                    output_tensor, 
                    [
                        (hidden_lstm_1, context_lstm_1),
                        ...., 
                        (hidden_list_n, context_lstm_n)
                    ]
                )
        """
        if self._lstm_count > 0:
            x, hx_cx_lst = x
        else:
            hx_cx_lst = []

        if hx_cx_lst is None:
            # initialise hx, cx to be zeros for all lstm's (account for batch size)
            zeros = lambda x : torch.zeros(self.lstm_hidden_size
                ) if x.ndim == 1 else torch.zeros(x.shape[0], self.lstm_hidden_size)
            hx_cx_lst = [(zeros(x), zeros(x)) for _ in range(self._lstm_count)]


        if len(hx_cx_lst) != self._lstm_count:
            raise ValueError(
                f"recurrent state list {hx_cx_lst} must have length {self._lstm_count} "
                f"not length {len(hx_cx_lst)}, as this is the number of lstm's in this network"
            )
        
        lstm_counter = 0
        new_hx_cx_lst = []
        for layer in self._layers:
            if isinstance(layer, nn.LSTMCell):
                hx, cx = layer(x, hx_cx_lst[lstm_counter])
                new_hx_cx_lst.append((hx, cx))
                x = hx
                lstm_counter += 1
            else:
                x = layer(x)

        if self._lstm_count == 0: 
            return x
        return x, new_hx_cx_lst
        

    def hard_update(self, source: nn.Module):
        """Updates the network parameters by copying the parameters of another network

        :param source (nn.Module): network to copy the parameters from
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source: nn.Module, tau: float):
        """Updates the network parameters with a soft update

        Moves the parameters towards the parameters of another network

        :param source (nn.Module): network to move the parameters towards
        :param tau (float): stepsize for the soft update
            (tau = 0: no update; tau = 1: copy parameters of source network)
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )


if __name__ == "__main__":
    # perform some simple tests on FCNetwork
    net = FCNetwork(
        [64, 128, 256], output_activation=nn.Tanh
    )
    assert isinstance(net._layers, nn.ModuleList)
    layers = nn.ModuleList([
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.Tanh()
    ])
    assert str(layers) == str(net._layers)

    net = FCNetwork(
        [64, 128, LSTM_STR, 256], 
        lstm_hidden_size=64,
        output_activation=nn.Tanh
    )
    layers = nn.ModuleList([
        nn.Linear(64, 128), nn.ReLU(),
        nn.LSTMCell(128, 64),
        nn.Linear(64, 256), nn.Tanh()
    ])
    assert str(layers) == str(net._layers)

    net = FCNetwork(
        [64, 128, LSTM_STR, LSTM_STR, 256], 
        lstm_hidden_size=64,
        output_activation=nn.Tanh
    )
    layers = nn.ModuleList([
        nn.Linear(64, 128), nn.ReLU(),
        nn.LSTMCell(128, 64),
        nn.LSTMCell(64, 64),
        nn.Linear(64, 256), nn.Tanh()
    ])
    assert str(layers) == str(net._layers)

    net = FCNetwork(
        [64, 128, LSTM_STR, LSTM_STR, 256, LSTM_STR], 
        lstm_hidden_size=64,
        output_activation=nn.Tanh
    )
    layers = nn.ModuleList([
        nn.Linear(64, 128), nn.ReLU(),
        nn.LSTMCell(128, 64),
        nn.LSTMCell(64, 64),
        nn.Linear(64, 256), nn.ReLU(),
        nn.LSTMCell(256, 64),
        nn.Tanh()
    ])
    assert str(layers) == str(net._layers)

    net = FCNetwork(
        [2, LSTM_STR], 
        lstm_hidden_size=64,
    )
    layers = nn.ModuleList([
        nn.LSTMCell(2, 64)
    ])
    assert str(layers) == str(net._layers)

    net = FCNetwork(
        [2, 32, LSTM_STR], 
        lstm_hidden_size=64,
    )
    layers = nn.ModuleList([
        nn.Linear(2, 32), nn.ReLU(),
        nn.LSTMCell(32, 64)
    ])
    assert str(layers) == str(net._layers)

    # test forward pass
    net = FCNetwork(
        [2, 32, LSTM_STR], 
        lstm_hidden_size=64,
    )
    
    out, hx_cx_lst = net((torch.zeros(2), None))
    assert out.shape == torch.Size([64])
    assert (out == hx_cx_lst[0][0]).all()
    assert hx_cx_lst[0][1].shape == torch.Size([64])

    out, hx_cx_lst = net((torch.zeros(2), hx_cx_lst))
    assert out.shape == torch.Size([64])
    assert (out == hx_cx_lst[0][0]).all()
    assert hx_cx_lst[0][1].shape == torch.Size([64])

    out, hx_cx_lst = net((torch.zeros(16, 2), None))
    assert out.shape == torch.Size([16, 64])
    assert (out == hx_cx_lst[0][0]).all()
    assert hx_cx_lst[0][1].shape == torch.Size([16, 64])

    out, hx_cx_lst = net((torch.zeros(16, 2), hx_cx_lst))
    assert out.shape == torch.Size([16, 64])
    assert (out == hx_cx_lst[0][0]).all()
    assert hx_cx_lst[0][1].shape == torch.Size([16, 64])

    net = FCNetwork(
        [2, 32, LSTM_STR, LSTM_STR], 
        lstm_hidden_size=64,
    )
    
    out, hx_cx_lst = net((torch.zeros(2), None))
    assert out.shape == torch.Size([64])
    assert (out == hx_cx_lst[-1][0]).all()
    assert hx_cx_lst[0][0].shape == torch.Size([64])
    assert hx_cx_lst[0][1].shape == torch.Size([64])
    assert hx_cx_lst[-1][1].shape == torch.Size([64])

    out, hx_cx_lst = net((torch.zeros(2), hx_cx_lst))
    assert out.shape == torch.Size([64])
    assert (out == hx_cx_lst[-1][0]).all()
    assert hx_cx_lst[0][0].shape == torch.Size([64])
    assert hx_cx_lst[0][1].shape == torch.Size([64])
    assert hx_cx_lst[-1][1].shape == torch.Size([64])

    out, hx_cx_lst = net((torch.zeros(16, 2), None))
    assert out.shape == torch.Size([16, 64])
    assert (out == hx_cx_lst[-1][0]).all()
    assert hx_cx_lst[0][0].shape == torch.Size([16, 64])
    assert hx_cx_lst[0][1].shape == torch.Size([16, 64])
    assert hx_cx_lst[-1][1].shape == torch.Size([16, 64])

    out, hx_cx_lst = net((torch.zeros(16, 2), hx_cx_lst))
    assert out.shape == torch.Size([16, 64])
    assert (out == hx_cx_lst[-1][0]).all()
    assert hx_cx_lst[0][0].shape == torch.Size([16, 64])
    assert hx_cx_lst[0][1].shape == torch.Size([16, 64])
    assert hx_cx_lst[-1][1].shape == torch.Size([16, 64])

    net = FCNetwork(
        [2, 1]
    )
    out = net(torch.zeros(2))
    assert out.shape == torch.Size([1])

    out = net(torch.zeros(16, 2))
    assert out.shape == torch.Size([16, 1])
