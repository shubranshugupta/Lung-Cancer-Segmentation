from pytorch_lightning.loggers import MLFlowLogger

class MyMLFlowLogger(MLFlowLogger):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @property
    def artifact_location(self) -> str:
        """Create the experiment if it does not exist to get the artifact location.

        Returns:
            The artifact location.
        """
        expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
        return expt.artifact_location