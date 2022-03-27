import os
import torch
import mlflow
from typing import Tuple, Dict, Any, Sequence
import logging as log
from mlflow.models import Model

from .utils import get_value
from .signature import Signature
from .logging import MyMLFlowLogger
from src import ROOT_FOLDER


def mlflow_setup(
    experiment_name:str,
    input_data_size:Sequence[int],
    output_data_size:Sequence[int],
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
    conda_file = os.path.join(ROOT_FOLDER, "conda.yaml")
    requirement_file = os.path.join(ROOT_FOLDER, "requirements.txt")
    mlf_logger.experiment.log_artifact(mlf_run_id, conda_file, "Model")
    mlf_logger.experiment.log_artifact(mlf_run_id, requirement_file, "Model")
    log.info("Saving Conda and Requirement files.")


    #Logging Hyperparameters
    for k, v in hyperparameters.items():
        mlf_logger.experiment.log_param(mlf_run_id, k, v)
    log.info("Saving Hyperparameters.")

    return mlf_logger, model_log, signature