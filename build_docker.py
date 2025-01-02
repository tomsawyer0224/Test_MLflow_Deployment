import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/aaa021a3ffaf491b84331ed1dcebf922/pyfunc_model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="pyfunc_docker",
    enable_mlserver=True
)