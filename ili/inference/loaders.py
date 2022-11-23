from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import numpy as np
import json
import pandas as pd
from summarizer.dataset import Dataset

class BaseLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """

class StaticNumpyLoader(BaseLoader):
    def __init__(
        self,
        in_dir: str,
        x_file: str,
        theta_file: str
    ):
        """Class to load single numpy files of summaries and parameters

        Args:
            in_dir (str): path to the location of stored data
            x_file (str): filename of the stored summaries
            theta_file (str): filename of the stored parameters
        """
        self.in_dir = Path(in_dir)
        self.x_path = self.in_dir / x_file
        self.theta_path = self.in_dir / theta_file

        self.x = np.load(self.x_path)
        self.theta = np.load(self.theta_path)

        if len(self.x) != len(self.theta):
            raise Exception('Stored summaries and parameters are not of same length.')

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return len(self.x)

    def get_all_data(self) -> np.array:
        """Returns all the loaded summaries

        Returns:
            np.array: summaries
        """
        return self.x

    def get_all_parameters(self):
        """Returns all the loaded parameters

        Returns:
            np.array: parameters
        """
        return self.theta

class SummarizerDatasetLoader(BaseLoader):
    def __init__(
        self,
        stage: str,
        data_dir: str,
        summary_root_file: str,
        param_file: str,
        train_test_split_file: str,
        param_names: List[str]
    ):
        """Class to load netCF files of summaries and a csv of parameters
        Basically a wrapper for ili-summarizer's Dataset, with added parameter loading

        Args:
            data_dir (str): path to the location of stored data
        """
        #self.num_nodes = num_nodes
        self.nodes = self.get_nodes_for_stage(stage=stage, train_test_split_file=train_test_split_file)
        self.data_dir = Path(data_dir)
        self.data = Dataset(
            nodes=self.nodes,
            path_to_data=self.data_dir,
            root_file=summary_root_file,
        )
        self.theta = self.load_parameters(
            data_dir = self.data_dir,
            param_file = param_file,
            nodes = self.nodes,
            param_names = param_names,
            
        )
        if len(self.data) != len(self.theta):
            raise Exception('Stored summaries and parameters are not of same length.')

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return len(self.nodes)

    def get_all_data(self) -> np.array:
        """Returns all the loaded summaries

        Returns:
            np.array: summaries
        """
        return self.data.load().reshape((self.num_nodes,-1))

    def get_all_parameters(self):
        """Returns all the loaded parameters

        Returns:
            np.array: parameters
        """
        return self.theta.values
    
    def get_nodes_for_stage(self, stage: str, train_test_split_file: str):
        with open(self.data_dir / train_test_split_file) as f:
            train_test_split = json.load(f)
        return train_test_split[stage]

    def load_parameters(self, data_dir: Path, param_file: str, nodes: List[int], param_names: List[str])->np.array:
        theta = pd.read_csv(
            data_dir / param_file,
            sep=' ',
            skipinitialspace=True
        ).iloc[nodes]
        return theta[param_names].values


# TODO: Add loaders which load dynamically from many files
