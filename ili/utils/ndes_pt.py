"""
Module to provide loading functions for ndes in various backends.

All Mixture Density Networks (mdn) have the configuration:
    hidden_features (int): width of hidden layers (each MDN has 3 hidden layers)
    num_components (int): number of Gaussian components in the mixture model

All flow-based models (maf, nsf, made) have the configuration:
    hidden_features (int): width of hidden layers in the coupling layers
    num_transforms (int): number of coupling layers

Linear classifiers (linear) have no arguments.

MLP and ResNet classifiers (mlp, resnet) have the configuration:
    hidden_features (int): width of hidden layers (each has 2 hidden layers)
"""

import logging

import numpy as np
import sbi
import torch
from torch import nn
import lampe
import zuko
from typing import List
from torch.distributions.transforms import Transform, identity_transform
from torch.distributions import biject_to, Distribution


def load_nde_sbi(
        engine: str,
        model: str,
        embedding_net: nn.Module = nn.Identity(),
        **model_args):
    """Load an nde from sbi.

    Args:
        engine (str): engine to use. 
            One of: NPE, NLE, NRE, SNPE, SNLE, or SNRE.
        model (str): model to use. 
            One of: mdn, maf, nsf, made, linear, mlp, resnet.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        **model_args: additional arguments to pass to the model.
    """
    # load NRE models (linear, mlp, resnet)
    if 'NRE' in engine:
        if model not in ['linear', 'mlp', 'resnet']:
            raise ValueError(f"Model {model} not implemented for {engine}.")
        return sbi.utils.classifier_nn(
            model=model, embedding_net_x=embedding_net, **model_args)

    if model not in ['mdn', 'maf', 'nsf', 'made']:
        raise ValueError(f"Model {model} not implemented for {engine}.")

    if (model == 'mdn'):
        # check for arguments
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
    else:
        # check for arguments
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")

    # Load NPE models (mdn, maf, nsf, made)
    if 'NPE' in engine:
        return sbi.utils.posterior_nn(
            model=model, embedding_net=embedding_net, **model_args)

    # Load NLE models (mdn, maf, nsf, made)
    if 'NLE' in engine:
        if embedding_net != nn.Identity():
            logging.warning(
                "Using an embedding_net with NLE models compresses theta, not "
                "x as might be expected.")
        return sbi.utils.likelihood_nn(
            model=model, embedding_net=embedding_net, **model_args)

    raise ValueError(f"Engine {engine} not implemented.")


class LampeNPE(nn.Module):
    """Simple wrapper to add an embedding network to an NPE model."""

    def __init__(
        self,
        nde: nn.Module,
        prior: Distribution,
        embedding_net: nn.Module = nn.Identity()
    ):
        super().__init__()
        self.nde = nde
        self.prior = prior
        self.embedding_net = embedding_net
        self.theta_transform = biject_to(prior.support)

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.nde(
            self.theta_transform.inv(theta),
            self.embedding_net(x))

    def flow(self, x: torch.Tensor):  # -> Distribution
        return self.nde.flow(self.embedding_net(x))

    def sample(
        self,
        shape: tuple,
        x: torch.Tensor,
        show_progress_bars: bool = False
    ) -> torch.Tensor:
        return self.theta_transform(self.flow(x).sample(shape)).cpu()


class LampeEnsemble(nn.Module):
    """Simple module to wrap an ensemble of NPE models."""

    def __init__(
        self,
        posteriors: List[LampeNPE],
        weights: torch.Tensor,
        device='cpu'
    ):
        super().__init__()
        self.posteriors = nn.ModuleList(posteriors)
        self.weights = weights
        assert len(self.posteriors) == len(self.weights)
        self.prior = self.posteriors[0].prior
        self.theta_transform = self.posteriors[0].theta_transform
        self._device = device

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            weight * npe(theta, x)
            for weight, npe in zip(self.weights, self.posteriors)
        ], dim=-1)

    def sample(
        self,
        shape: tuple,
        x: torch.Tensor,
        show_progress_bars: bool = True
    ):
        num_samples = np.prod(shape)
        per_model = torch.round(
            num_samples * self.weights/self.weights.sum()).numpy().astype(int)
        if show_progress_bars:
            logging.info(f"Sampling models with {per_model} samples each.")
        samples = torch.cat([
            nde.sample((N,), x, show_progress_bars=show_progress_bars)
            for nde, N in zip(self.posteriors, per_model)
        ], dim=0)
        return samples.reshape(*shape, -1)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.forward(theta, x).sum(dim=-1).detach()


def load_nde_lampe(
        model: str,
        embedding_net: nn.Module = nn.Identity(),
        ** model_args):
    """Load an nde from lampe.

    Args:
        model (str): model to use.
            One of: mdn, maf, nsf, made, linear, mlp, resnet.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        **model_args: additional arguments to pass to the model.
    """
    if model == 'mdn':
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        model_args['hidden_features'] = [model_args['hidden_features']] * 3
        model_args['components'] = model_args.pop('num_components', 2)
        flow_class = zuko.flows.mixture.GMM
    else:
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        model_args['hidden_features'] = [
            model_args['hidden_features']] * 2
        model_args['transforms'] = model_args.pop('num_transforms', 2)

    if model == 'maf':
        flow_class = zuko.flows.autoregressive.MAF

    def net_constructor(x_batch, theta_batch, prior):
        z_batch = embedding_net(x_batch)
        z_shape = z_batch.shape[1:]
        theta_shape = theta_batch.shape[1:]

        if (len(z_shape) > 1):
            raise ValueError("Embedding network must return a vector.")
        if (len(theta_shape) > 1):
            raise ValueError("Parameters theta must be a vector.")

        nde = lampe.inference.NPE(
            theta_dim=theta_shape[0],
            x_dim=z_shape[0],
            build=flow_class,
            **model_args
        )
        return LampeNPE(
            nde=nde,
            embedding_net=embedding_net,
            prior=prior
        )

    return net_constructor