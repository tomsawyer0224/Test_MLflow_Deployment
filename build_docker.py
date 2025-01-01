import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/15ce1f0945744ad58b9f8db503a3323c/pyfunc_model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="pyfunc_docker",
    enable_mlserver=True
)