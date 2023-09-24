"""
Abstract class for load the dataset

:author: Ricardo Espantaleón Pérez
"""


class BaseMakeDataset:
    def __init__(self, config):
        """
        Parametrized constructor to load the configuration file of the dataset

        :param config: config file where to load all params for load a specific dataset
        """
        self.config = config

    def pre_process_dataset(self):
        """
        Method to pre-process the dataset

        :return: pre-processed dataset
        """
        raise NotImplementedError

    def get_training_data(self):
        """
        Method to load the training data

        :return: training data
        """
        raise NotImplementedError

    def get_validation_data(self):
        """
        Method to load the validation data

        :return: validation data
        """
        raise NotImplementedError
