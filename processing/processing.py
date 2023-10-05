
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

    def extractComplexType(self):

        df_temp = self.df2.apply(lambda row: pd.Series(row['cast']), axis = 1).stack().reset_index(level = 1, drop = True)
        df_temp = pd.DataFrame(df_temp, columns = ['cast'])

        dff = df_temp.to_dict(orient = 'list')

        new_dataframe_cast = pd.DataFrame({
            'cast_id': [di['cast_id'] for di in dff],
            'character': [di['character'] for di in dff],
            'credit_id': [di['credit_id'] for di in dff],
            'gender': [di['cast_id'] for di in dff],
            'id': [di['id'] for di in dff],
            'name': [di['name'] for di in dff],
            'order': [di['order'] for di in dff]
        })






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