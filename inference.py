# Prerequisite: serve a custom pyfunc OpenAI model (not mlflow.openai) on localhost:5678
#   that defines inputs in the below format and params of `temperature` and `max_tokens`

import json
import requests
import numpy as np
import pandas as pd
from mlflow.models import convert_input_example_to_serving_input
from data_module import Data_Module

exam_path = "./serving_input_example.json"
with open(exam_path, "r") as f:
    payload_dict = json.load(f)
payload = json.dumps(payload_dict)
# payload = convert_input_example_to_serving_input(
#     pd.DataFrame(np.random.rand(4,28*28))
# )
response = requests.post(
    url=f"http://127.0.0.1:5001/invocations",
    data=payload,
    headers={"Content-Type": "application/json"},
)
print(response.json())

datasets = Data_Module()
test_data, test_target = datasets.test_dataset
inference_payload = convert_input_example_to_serving_input(test_data)

response = requests.post(
    url=f"http://127.0.0.1:5001/invocations",
    data=inference_payload,
    headers={"Content-Type": "application/json"},
)
print(response.json())
print(test_target)
