
from processing.processing import ProcessingDf
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    df = pd.read_csv('./archive/tmdb_5000_movies.csv')
    df1 = pd.read_csv('./archive/tmdb_5000_credits.csv')

    print(df1["title"])




    """dataset_unione = ProcessingDf('./archive/tmdb_5000_movies.csv', './archive/tmdb_5000_credits.csv')
    chiavi_di_unione = ['id', 'movie_id']  # Lista di chiavi per l'unione
    risultato_unione = dataset_unione.unisci(chiavi_di_unione)

    # Stampa il risultato dell'unione
    if risultato_unione is not None:
        print(risultato_unione.head())"""

