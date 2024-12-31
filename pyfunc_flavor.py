import mlflow

class MLflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)