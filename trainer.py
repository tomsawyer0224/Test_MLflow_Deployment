import mlflow
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from mlflow.models import infer_signature
from data_module import Data_Module
from pyfunc_flavor import MLflowModel

class Trainer:
    def __init__(self, use_pyfunc_flavor = False):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
        self.experiment_id = self._get_or_create_experiment("Test_MLflow_Deployment")
        self.datasets = Data_Module()
        self.model = SGDClassifier()
        self.use_pyfunc_flavor = use_pyfunc_flavor
    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """
        gets an existing experiment or creates a new experiment
        args:
            experiment_name: the name of experiment
            client: an instance of MlflowClient
        returns:
            experiment ID
        """
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)
    def train(self):
        train_data, train_target = self.datasets.train_dataset
        input_example = train_data[:2]
        signature = infer_signature(train_data[:2], train_target[:2])
        run_name = f"model_at_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name):
            self.model.fit(train_data, train_target)
            if not self.use_pyfunc_flavor:
                mlflow.sklearn.log_model(
                    self.model,
                    artifact_path = "sklearn_model",
                    signature = signature,
                    input_example = input_example
                )
            else:
                mlflow.pyfunc.log_model(
                artifact_path = "pyfunc_model",
                python_model = MLflowModel(self.model),
                signature = signature,
                input_example = input_example
                )
    def test(self):
        pass
if __name__=="__main__":
    trainer_sklearn = Trainer(use_pyfunc_flavor=False)
    trainer_sklearn.train()

    trainer_pyfunc = Trainer(use_pyfunc_flavor=True)
    trainer_pyfunc.train()