
import pandas as pd

class ProcessingDf:

    def __init__(self, dataset):
        self.dataset = dataset
        df = pd.read_csv(dataset)

