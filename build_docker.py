import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/b245d5a09e6d43f68b3ff7b73b79a0eb/sklearn_model"
mlflow.models.build_docker(
    model_uri=model_uri, name="sklearn_docker", enable_mlserver=True
)
