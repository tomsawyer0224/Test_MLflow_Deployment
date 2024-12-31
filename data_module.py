from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Data_Module:
    def __init__(self):
        dataset = load_iris(as_frame=True)
        self.class_names = dataset["target_names"]
        data = dataset["data"]
        target = dataset["target"].apply(lambda i: str(self.class_names[i]))
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
    @property
    def train_dataset(self):
        return self.train_data, self.train_target
    @property
    def test_dataset(self):
        return self.test_data, self.test_target
    
if __name__=="__main__":
    dm = Data_Module()
    train_data, train_target = dm.train_dataset
    test_data, test_target = dm.test_dataset
    print(f"{train_data=}\n")
    print(f"{train_target=}\n")
    print(f"{test_data=}\n")
    print(f"{test_target=}\n")