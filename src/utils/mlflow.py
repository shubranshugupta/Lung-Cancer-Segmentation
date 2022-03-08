import os
import torch
import mlflow
from typing import Tuple, Dict, Any
import logging as log
from mlflow.models import Model

from .utils import get_value
from .signature import Signature
from .logging import MyMLFlowLogger


def mlflow_setup(
    experiment_name:str,
    input_data_size:Tuple[int],
    output_data_size:Tuple[int],
    hyperparameters:Dict[str, Any],
) -> Tuple[MyMLFlowLogger, Model, Signature]:
    """
    A function to setup mlflow

    :param experiment_name: experiment name
    :param input_data_size: input data size
    :param output_data_size: output data size
    :param hyperparameters: hyperparameters

    :return: mlflow logger, mlflow model, mlflow signature

    Example:
    ```python
    mlflow_setup(
        experiment_name="test",
        input_data_size=(1, 28, 28),
        output_data_size=(10,),
        hyperparameters={
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "device": "cpu",
            }
    )
    ```
    """

    log.info("Initializing MLFlow Logger..")
    tracking_uri = get_value("config.yaml", "MLFLOW_TRACKING_URI")
    log.info(f"MLFlow Experiment: {experiment_name}, MLFlow Tracking URI: {tracking_uri}")
    mlf_logger = MyMLFlowLogger(experiment_name=experiment_name, 
                                tracking_uri=tracking_uri)
    mlf_run_id = mlf_logger.run_id

    # Defining Signature
    sample_inputs = torch.rand(input_data_size)
    sample_outputs = torch.rand(output_data_size)
    signature = Signature(sample_inputs, sample_outputs)

    # Defining Mlflow Model saving
    log.info("Initializing MLFlow Model saving class..")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_id=mlf_run_id)
    model_log = Model(artifact_path=mlf_logger.artifact_location, 
                        run_id=mlf_run_id, flavors=mlflow.pytorch)

    # Logging Environment detail
    conda_file = os.path.join(os.getcwd(), "conda_env.yml")
    requirement_file = os.path.join(os.getcwd(), "requirements.txt")
    mlf_logger.experiment.log_artifact(mlf_run_id, conda_file, "Model")
    mlf_logger.experiment.log_artifact(mlf_run_id, requirement_file, "Model")
    log.info("Saving Conda and Requirement files.")

    # Hyperparameters
    # epoch = 10
    # batch_size = 32
    # test_size = 0.2
    # dropout = 0.5
    # lr = 1e-3
    # batchnorm = False

    #Logging Hyperparameters
    mlf_logger.experiment.log_hyperparams(hyperparameters)
    # mlf_logger.experiment.log_param(mlf_run_id, "epoch", epoch)
    # mlf_logger.experiment.log_param(mlf_run_id, "batch_size", batch_size)
    # mlf_logger.experiment.log_param(mlf_run_id, "test_size", test_size)
    # mlf_logger.experiment.log_param(mlf_run_id, "dropout", dropout)
    # mlf_logger.experiment.log_param(mlf_run_id, "lr", lr)
    # mlf_logger.experiment.log_param(mlf_run_id, "batchnorm", batchnorm)
    log.info("Saving Hyperparameters.")

    return mlf_logger, model_log, signature