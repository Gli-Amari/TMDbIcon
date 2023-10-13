from datetime import datetime
from processing.TMDbAPI import TMDbAPI
from processing.Proccessing import Processing
import pandas as pd
from pathlib import Path


def checking_path(path_csv):
    check_path = Path(path_csv)
    if check_path.is_dir():
        complete_path = Path(path_csv + "Popular_film.csv")
        if complete_path.exists():
            return True
        else:
            return False


# richiamo in una routine esclusivamente le op. per fare processing su dataset Popular_film!
def processingPopularFilmDataset(movie_dataframe):
    processing = Processing(movie_dataframe)

    # sistemiamo le feature che hanno al loro interno piu valori (un film puo aver piu generi per ese.)
    processing.extraction(col_name='genres', key='name')
    processing.extraction(col_name='production_companies', key='name')
    processing.extraction(col_name='production_countries', key='iso_3166_1')
    processing.extraction(col_name='spoken_languages', key='iso_639_1')



    movie_dataframe = movie_dataframe.dropna() #cancella righe che contengono NaNg
    #movie_dataframe = movie_dataframe.drop_duplicates(keep= 'first') #mantiene solo la prima occorrenza dei duplicati

    return movie_dataframe


if __name__ == "__main__":

    api_key = '293e12b22f35ee4b22ee998909252150'
    endpoint = "movie/popular"  # Esempio: film popolari
    max_pages = 2  # Numero massimo di pagine da ottenere
    path_csv = "./dataset/"

    # estrazione dati da server TMDb
    TMDbAPI = TMDbAPI(api_key)

    complete_path = path_csv + "Popular_film.csv"
    if checking_path(path_csv) is False:
        movie_dataframe = TMDbAPI.fetch_movie_data(endpoint, max_pages)

        # processing dataset Popular_film
        processingPopularFilmDataset(movie_dataframe)

        movie_dataframe.to_csv(complete_path, index=False)

    df = pd.read_csv("./dataset/Popular_film.csv")
    print(df.info())
    #print(df['production_companies'])






