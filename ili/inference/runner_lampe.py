"""
Module to train posterior inference models using the sbi package
"""

import json
import yaml
import time
import logging
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lampe
from lampe.inference import NPE, NPELoss
from pathlib import Path
from typing import Dict, List, Callable, Optional
from torch.distributions import Independent
from ili.dataloaders import _BaseLoader
from ili.utils import load_class, load_from_config, load_nde_sbi, LampeEnsemble

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_sbi.yaml"
)


class LampeRunner():
    """Class to train posterior inference models using the sbi package

    Args:
        prior (Independent): prior on the parameters
        inference_class (NeuralInference): sbi inference class used to
            train neural posteriors
        nets (List[Callable]): list of neural nets for amortized posteriors,
            likelihood models, or ratio classifiers
        embedding_net (nn.Module): neural network to compress high
            dimensional data into lower dimensionality
        train_args (Dict): dictionary of hyperparameters for training
        out_dir (Path): directory where to store outputs
        proposal (Independent): proposal distribution from which existing
            simulations were run, for single round inference only. By default,
            sbi will set proposal = prior unless a proposal is specified.
            While it is possible to choose a prior on parameters different
            than the proposal for SNPE, we advise to leave proposal to None
            unless for test purposes.
        name (str): name of the model (for saving purposes)
        signatures (List[str]): list of signatures for each neural net
    """

    def __init__(
        self,
        prior: Independent,
        nets: List[Callable],
        train_args: Dict = {},
        out_dir: Path = None,
        device: str = 'cpu',
        embedding_net: nn.Module = None,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = None,
    ):
        self.prior = prior
        self.train_args = train_args
        self.device = device
        self.name = name
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.nets = nets
        self.embedding_net = embedding_net
        self.signatures = signatures
        if self.signatures is None:
            self.signatures = [""]*len(self.nets)
        self.train_args = dict(
            batch_size=32, learning_rate=1e-3,
            stop_after_epochs=30, clip_max_norm=10)
        self.train_args.update(train_args)

    # @classmethod
    # def from_config(cls, config_path: Path, **kwargs) -> "SBIRunner":
    #     """Create an sbi runner from a yaml config file

    #     Args:
    #         config_path (Path, optional): path to config file
    #         **kwargs: optional keyword arguments to overload config file
    #     Returns:
    #         SBIRunner: the sbi runner specified by the config file
    #     """
    #     with open(config_path, "r") as fd:
    #         config = yaml.safe_load(fd)

    #     # optionally overload config with kwargs
    #     config.update(kwargs)

    #     # load prior distribution
    #     config['prior']['args']['device'] = config['device']
    #     prior = load_from_config(config["prior"])

    #     # load proposal distributions
    #     proposal = None
    #     if "proposal" in config:
    #         config['proposal']['args']['device'] = config['device']
    #         proposal = load_from_config(config["proposal"])

    #     # load embedding net
    #     if "embedding_net" in config:
    #         embedding_net = load_from_config(
    #             config=config["embedding_net"],
    #         )
    #     else:
    #         embedding_net = nn.Identity()

    #     # load logistics
    #     train_args = config["train_args"]
    #     out_dir = Path(config["out_dir"])
    #     if "name" in config["model"]:
    #         name = config["model"]["name"]+"_"
    #     else:
    #         name = ""
    #     signatures = []
    #     for type_nn in config["model"]["nets"]:
    #         signatures.append(type_nn.pop("signature", ""))

    #     # load inference class and neural nets
    #     inference_class = load_class(
    #         module_name=config["model"]["module"],
    #         class_name=config["model"]["class"],
    #     )
    #     nets = [load_nde_sbi(config['model']['class'],
    #                          embedding_net=embedding_net,
    #                          **model_args)
    #             for model_args in config['model']['nets']]

    #     # initialize
    #     return cls(
    #         prior=prior,
    #         proposal=proposal,
    #         inference_class=inference_class,
    #         nets=nets,
    #         device=config["device"],
    #         embedding_net=embedding_net,
    #         train_args=train_args,
    #         out_dir=out_dir,
    #         signatures=signatures,
    #         name=name,
    #     )
    def _train_epoch(self, model, loader_train, loader_val, stepper):
        """Train a single epoch of a neural network model."""
        loss = NPELoss(model)
        model.train()

        loss_train = torch.stack([
            stepper(loss(theta, x))
            for x, theta in loader_train
        ]).mean().item()

        model.eval()
        with torch.no_grad():
            loss_val = torch.stack([
                loss(theta, x)
                for x, theta in loader_val
            ]).mean().item()

        return loss_train, loss_val

    def _train_round(self, models: List[Callable],
                     x: torch.Tensor, theta: torch.Tensor):
        """Train a single round of inference for an ensemble of models."""
        # split data into train and validation
        mask = torch.randperm(x.shape[0]) < int(0.9*x.shape[0])
        x_train, x_val = x[mask], x[~mask]
        theta_train, theta_val = theta[mask], theta[~mask]

        data_train = TensorDataset(x_train, theta_train)
        data_val = TensorDataset(x_val, theta_val)
        loader_train = DataLoader(
            data_train, batch_size=self.train_args["batch_size"])
        loader_val = DataLoader(
            data_val, batch_size=self.train_args["batch_size"])

        posteriors, summaries = [], []
        for i, model in enumerate(models):
            logging.info(f"Training model {i+1} / {len(models)}.")

            # initialize model
            x_, y_ = next(iter(loader_train))
            model = model(x_, y_, self.prior).to(self.device)

            # define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.train_args["learning_rate"]
            )
            stepper = lampe.utils.GDStep(
                optimizer, clip=self.train_args["clip_max_norm"])

            # train model
            best_val = float('inf')
            wait = 0
            summary = {'training_log_probs': [], 'validation_log_probs': []}
            with tqdm(iter(range(10000)), unit=' epochs') as tq:
                for epoch in tq:
                    loss_train, loss_val = self._train_epoch(
                        model=model,
                        loader_train=loader_train,
                        loader_val=loader_val,
                        stepper=stepper,
                    )
                    tq.set_postfix(
                        loss=loss_train,
                        loss_val=loss_val,
                    )
                    summary['training_log_probs'].append(-loss_train)
                    summary['validation_log_probs'].append(-loss_val)

                    # check for convergence
                    if loss_val < best_val:
                        best_val = loss_val
                        best_model = model.state_dict()
                        wait = 0
                    elif wait > self.train_args["stop_after_epochs"]:
                        break
                    else:
                        wait += 1
                else:
                    logging.warning("Training did not converge in 10k epochs.")
                summary['best_validation_log_prob'] = -best_val
                summary['epochs_trained'] = epoch

            # save model
            model.load_state_dict(best_model)
            posteriors.append(model)
            summaries.append(summary)

        # ensemble all trained models, weighted by validation loss
        val_logprob = torch.tensor(
            [float(x["best_validation_log_prob"]) for x in summaries]
        ).to(self.device)
        # Exponentiate with numerical stability
        weights = torch.exp(val_logprob - val_logprob.max())
        weights /= weights.sum()

        posterior_ensemble = LampeEnsemble(
            posteriors, weights,
            device=self.device)

        # record the name of the ensemble
        posterior_ensemble.name = self.name
        posterior_ensemble.signatures = self.signatures

        return posterior_ensemble, summaries

    # def _save_models(self, posterior_ensemble: NeuralPosteriorEnsemble,
    #                  summaries: List[Dict]):
    #     """Save models to file."""

    #     logging.info(f"Saving model to {self.out_dir}")
    #     str_p = self.name + "posterior.pkl"
    #     str_s = self.name + "summary.json"
    #     with open(self.out_dir / str_p, "wb") as handle:
    #         pickle.dump(posterior_ensemble, handle)
    #     with open(self.out_dir / str_s, "w") as handle:
    #         json.dump(summaries, handle)

    def __call__(self, loader: _BaseLoader, seed: int = None):
        """Train your posterior and save it to file

        Args:
            loader (_BaseLoader): dataloader with stored data-parameter pairs
            seed (int): torch seed for reproducibility
        """

        # set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # setup training engines for each model in the ensemble
        logging.info("MODEL INFERENCE CLASS: NPE")

        # load single-round data
        x = torch.Tensor(loader.get_all_data()).to(self.device)
        theta = torch.Tensor(loader.get_all_parameters()).to(self.device)

        # train a single round of inference
        t0 = time.time()
        posterior_ensemble, summaries = self._train_round(
            models=self.nets,
            x=x,
            theta=theta,
        )
        logging.info(f"It took {time.time() - t0} seconds to train models.")

        # save if output path is specified
        if self.out_dir is not None:
            self._save_models(posterior_ensemble, summaries)

        return posterior_ensemble, summaries
