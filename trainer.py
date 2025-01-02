import mlflow
from mlflow.models import convert_input_example_to_serving_input
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from mlflow.models import infer_signature, validate_serving_input
import pandas as pd
import os
import json
from data_module import Data_Module
from pyfunc_flavor import MLflowModel

class Trainer:
    def __init__(self, use_pyfunc_flavor = False):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
        self.experiment_id = self._get_or_create_experiment("Test_MLflow_Deployment")
        self.run_name = f"model_at_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.run_id = None
        self.datasets = Data_Module()
        self.model = SGDClassifier()
        self.use_pyfunc_flavor = use_pyfunc_flavor
    @property
    def model_uri(self):
        model_name = "pyfunc_model" if self.use_pyfunc_flavor else "sklearn_model"
        model_uri = f"runs:/{self.run_id}/{model_name}"
        return model_uri
    @property
    def serving_payload(self):
        #model_name = "pyfunc_model" if self.use_pyfunc_flavor else "sklearn_model"
        # json_path = os.path.join(
        #     "./mlartifacts", self.experiment_id, self.run_id, "artifacts", model_name, "serving_input_example.json"
        # )
        json_path = "./serving_input_example.json"
        with open(json_path, "r") as f:
            serving_input_example = json.load(f)
        payload = json.dumps(serving_input_example)
        # payload = convert_input_example_to_serving_input(
        #     pd.DataFrame()
        # )
        return payload
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
        #run_name = f"model_at_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name) as run:
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
                input_example = input_example,
                infer_code_paths = True
                )
        self.run_id = run.info.run_id
    def test(self):
        serving_payload = self.serving_payload
        # Validate the serving payload works on the model
        print("validate")
        predictions = validate_serving_input(self.model_uri, serving_payload)
        print(predictions)
        print("-"*30)
        print("inference")
        test_data, test_target = self.datasets.test_dataset
        loaded_model = mlflow.pyfunc.load_model(self.model_uri)
        preds = loaded_model.predict(test_data)
        print(preds)
        print(f"score = {self.model.score(test_data, test_target)}")
if __name__=="__main__":
    trainer_sklearn = Trainer(use_pyfunc_flavor=False)
    trainer_sklearn.train()
    trainer_sklearn.test()
    
    trainer_pyfunc = Trainer(use_pyfunc_flavor=True)
    trainer_pyfunc.train()
    trainer_pyfunc.test()