import yaml
import time
import logging
import pickle
import torch
import torch.nn as nn
import sbi
from pathlib import Path
from typing import Dict, Any, List, Callable
from torch.distributions import Independent
from sbi.inference import NeuralInference
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from ili.utils import load_class, load_from_config

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_ensemble.yaml"
)


class SBIRunner:
    def __init__(
        self,
        prior: Independent,
        inference_class: NeuralInference,
        neural_density_estimators: List[Callable],
        device: str,
        embedding_net: nn.Module,
        train_args: Dict,
        output_path: Path,
    ):
        """Class to train posterior inference models using the sbi package

        Args:
            prior (Independent): prior on the parameters
            inference_class (NeuralInference): sbi inference class used to that train neural posteriors
            neural density estimators (List[Callable]): list of neural density estimators to train
            embedding_net (nn.Module): neural network to compress high dimensional data into lower dimensionality
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.prior = prior
        self.inference_class = inference_class
        self.neural_density_estimators = neural_density_estimators 
        self.device = device
        self.embedding_net = embedding_net
        self.train_args = train_args
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path) -> "SBIRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        prior = load_from_config(config["prior"])
        if "embedding_net" in config:
            embedding_net = load_from_config(
                config=config["embedding_net"],
            )
        else:
            embedding_net = nn.Identity()
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        neural_density_estimators = cls.load_neural_density_estimators(
            embedding_net=embedding_net,
            density_estimator_config=config["model"]["neural_density_estimators"],
            posterior=True if config["model"]["class"] == "SNPE" else False,
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            prior=prior,
            inference_class=inference_class,
            neural_density_estimators=neural_density_estimators,
            device=config["device"],
            embedding_net=embedding_net,
            train_args=train_args,
            output_path=output_path,
        )


    @classmethod
    def load_neural_density_estimators(
        cls,
        embedding_net: nn.Module,
        density_estimator_config: List[Dict],
        posterior: bool = False,
    ) -> List[Callable]:
        """Load the inference model

        Args:
            embedding_net (nn.Module): neural network to compress high dimensional data
            density_estimator_config (List[Dict]): list with configurations for each neural density estimator 
            model in the ensemble
            posterior (bool, optional): whether to load a posterior or likelihood model. Defaults to True.

        Returns:
            List[Callable]: list of neural density estiamtor models with forward methods
        """
        neural_des = []
        for model_args in density_estimator_config:
            if posterior:
                density_estimator = sbi.utils.posterior_nn(
                    embedding_net=embedding_net,
                    **model_args,
                )
            else:
                density_estimator = sbi.utils.likelihood_nn(
                        embedding_net=embedding_net,
                        **model_args,
                    )
            neural_des.append(density_estimator)
        return neural_des 

    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
        """

        t0 = time.time()
        x = torch.Tensor(loader.get_all_data())
        theta = torch.Tensor(loader.get_all_parameters())
        posteriors, val_loss = [], []
        for n, density_estimator in enumerate(self.neural_density_estimators):
            logging.info(
                f"Training model {n+1} out of {len(self.neural_density_estimators)} ensemble models"
            )
            model = self.inference_class(
                prior=self.prior,
                density_estimator=density_estimator,
                device=self.device,
            )
            model = model.append_simulations(theta, x)
            if not isinstance(self.embedding_net, nn.Identity):
                self.embedding_net.initalize_model(n_input=x.shape[-1])
            de = model.train(
                **self.train_args,
            )
            # Store trained density estimator 
            with open(self.output_path / f"density_estimator_{n}.pkl", "wb") as handle:
                pickle.dump(de, handle)
            posteriors.append(
                model.build_posterior(
                   de,
                   sample_with='vi' if 'snle' in str(self.inference_class) else None,
                   vi_method="fKL" if 'snle' in str(self.inference_class) else None,
            ))
            val_loss += model.summary["best_validation_log_prob"]
        posterior = NeuralPosteriorEnsemble(
            posteriors=posteriors,
            weights=torch.tensor([float(vl) for vl in val_loss]),
        )
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
        logging.info(f"It took {time.time() - t0} seconds to train all models.")
