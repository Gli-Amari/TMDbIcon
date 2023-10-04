
import pandas as pd

class ProcessingDf:

    def __init__(self, ds1, ds2):
        self.df1 = ds1
        self.df2 = ds2

    """
        nel try-catch prima di mergiare devono essere sistemate le feature 
    """
    def mergingDf(self, chiavi):

        try:
            risultato = pd.merge(self.df1, self.df2,
                                 on = chiavi, how = 'inner')
            return risultato
        except Exception as e:
            print(f"Si è verificato un errore durante l'unione dei dataset: {str(e)}")
            return None

    def convertObjectType(self, df):

        for colonna in df.columns:
            # Utilizza il metodo pd.to_numeric() per convertire in numerico se possibile
            df[colonna] = pd.to_numeric(df[colonna], errors = 'ignore')
            # Utilizza il metodo pd.to_datetime() per convertire in data se possibile
            try:
                df[colonna] = pd.to_datetime(df[colonna])
            except ValueError:
                pass  # Se non è possibile convertire in data, continua con il tipo 'object'
        return df