from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import torch

class Signature:
    def __init__(self, inputs:torch.Tensor, outputs: torch.Tensor) -> None:
        if isinstance(inputs, torch.Tensor) and isinstance(outputs, torch.Tensor):
            self.inputs = inputs.numpy()
            self.outputs = outputs.numpy()

            self._signature = self.__get_signature()
        else:
            raise ValueError("Inputs and outputs must be torch.Tensor")
    
    def __get_signature(self) -> ModelSignature:
        input_spec = TensorSpec(type=self.inputs.dtype, shape=self.inputs.shape, name="input")
        output_spec = TensorSpec(type=self.outputs.dtype, shape=self.outputs.shape, name="output")

        input_schem = Schema([input_spec])
        output_schem = Schema([output_spec])

        return ModelSignature(inputs=input_schem, outputs=output_schem)
    
    @property
    def return_signature(self) -> ModelSignature:
        return self._signature
    
    @property
    def shape(self) -> tuple:
        return self.inputs.shape, self.outputs.shape
