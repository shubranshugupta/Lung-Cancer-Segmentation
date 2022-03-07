from pytorch_lightning.callbacks import ModelCheckpoint
from mlflow.models import Model
from pytorch_lightning.utilities import rank_zero_only
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
import pickle
import mlflow.pytorch
import torch
from .signature import Signature
from torch.jit import trace

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, 
                 mlflow_model:Model, 
                 signature:Signature, 
                 register_model_name:str=None, 
                 *args, **kwargs) -> None:
        """
        :param mlflow_model: MLflow model object
        :param register_model_name: Name of the model to register
        :param signature: Model signature
        """
        super().__init__(*args, **kwargs)
        self.mlflow_model = mlflow_model
        self.register_model_name = register_model_name
        self.signature = signature

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        shape = self.signature.shape[0]
        if torch.cuda.is_available():
            example_inputs = torch.rand(shape).to(torch.device('cuda'))
        else:
            example_inputs = torch.rand(shape).to(torch.device('cpu'))
        model = trace(pl_module, example_inputs=example_inputs)
        self.mlflow_model.log(artifact_path="Model",
                              flavor=mlflow.pytorch,
                              pytorch_model=model,
                              pickle_module=pickle,
                              registered_model_name=self.register_model_name,
                              signature=self.signature.return_signature,
                              await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS)