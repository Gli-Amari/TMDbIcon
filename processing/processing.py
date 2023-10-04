
import pandas as pd

class ProcessingDf:

    def __init__(self, *datasets):
        self.datasets = datasets

    def unisci(self, chiavi):

        if len(self.datasets) < 2:
            print("Devi fornire almeno due dataset da unire.")
            return None
        try:
            dfs = [pd.read_csv(dataset, delimiter='\t') for dataset in self.datasets]

            risultato = dfs[0]
            for i in range(1, len(dfs)):
                for chiave in chiavi:
                    risultato = pd.merge(risultato, dfs[i], on=chiave, how='inner')
            return risultato
        except Exception as e:
            print(f"Si è verificato un errore durante l'unione dei dataset: {str(e)}")
            return None
