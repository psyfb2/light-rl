import torch

from torch.nn import Module, Parameter 

class VirtualBatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float=1e-5):
        """ Virtual batch normalisation. Apply before
        activation for MLP.

        Args:
            num_features (int): number of output features of prev layer
            eps (float, optional): epsilon for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps 
        self.ref_mean = self.register_parameter('ref_mean', None)
        self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)

        # define gamma and beta parameters
        gamma = torch.normal(mean=torch.ones(1, num_features), std=0.02)
        self.gamma = Parameter(gamma.float())
        self.beta = Parameter(torch.FloatTensor(1, num_features).fill_(0))

    def get_stats(self, x: torch.Tensor):
        """ Calculates mean and mean square for given batch x.

        Args:
            x (torch.Tensor): tensor containing batch of activations

        Returns:
            mean (torch.Tensor): mean tensor over features
            mean_sq (torch.Tensor): squared mean tensor over features
        """
        mean = x.mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x: torch.Tensor, ref_mean: torch.Tensor, ref_mean_sq: torch.Tensor):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        The input parameter is_reference should indicate whether it is a forward pass
        for reference batch or not.

        Args:
            x (torch.Tensor): input tensor
            ref_mean (torch.Tensor): mean of reference batch obtained from
                first call to forward. If this is first call pass None
            ref_mean_sq (torch.Tensor): mean squared of reference batch obtained from
                first call to forward. If this is first call pass None
        Result:
            x (torch.Tensor): normalized output
            ref_mean (torch.Tensor): mean used for normalization
            ref_mean_sq (torch.Tensor): mean squarred used for normalization

        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self._normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self._normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, mean_sq: torch.Tensor):
        """ Normalize tensor x given the statistics.

        Args:
            x (torch.Tensor): input tensor
            mean (torch.Tensor): mean over features. it has size [1:num_features:]
            mean_sq (torch.Tensor): squared means over features.

        Result:
            x (torch.Tensor): normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        if mean.size(1) != self.num_features:
            raise Exception(
                    'Mean size not equal to number of featuers : given {}, expected {}'
                    .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                    'Squared mean tensor size not equal to number of features : given {}, expected {}'
                    .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))