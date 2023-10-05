
from processing.processing import ProcessingDf
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    df = pd.read_csv('./archive/tmdb_5000_movies.csv')
    df1 = pd.read_csv('./archive/tmdb_5000_credits.csv')

    df_temp = df1.apply(lambda row: pd.Series(row['cast']), axis=1).stack().reset_index(level=1,drop=True)
    df_temp = pd.DataFrame(df_temp, columns=['cast'])

    dff = df_temp.to_dict(orient = 'list')

    for chiave, valore in dff.items():
        try:
            dff[chiave] = int(valore)
        except (ValueError, TypeError):
            # Se non è possibile convertire il valore in intero, mantieni la stringa originale
            pass

print(dff['cast_id'])

    new_dataframe_cast = pd.DataFrame({
        'cast_id': [di['cast_id'] for di in dff],
        'character': [di['character'] for di in dff],
        'credit_id': [di['credit_id'] for di in dff],
        'gender': [di['cast_id'] for di in dff],
        'id': [di['id'] for di in dff],
        'name': [di['name'] for di in dff],
        'order': [di['order'] for di in dff]
    })

    print(df_temp.info())
    print("\n")
    print(new_dataframe_cast)
    print("\n")
    print(dff)



    """dataset_unione = ProcessingDf('./archive/tmdb_5000_movies.csv', './archive/tmdb_5000_credits.csv')
    chiavi_di_unione = ['id', 'movie_id']  # Lista di chiavi per l'unione
    risultato_unione = dataset_unione.unisci(chiavi_di_unione)

    # Stampa il risultato dell'unione
    if risultato_unione is not None:
        print(risultato_unione.head())"""

