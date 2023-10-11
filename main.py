import os

from processing.TMDbAPI import TMDbAPI
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


if __name__ == "__main__":

    api_key = '293e12b22f35ee4b22ee998909252150'
    endpoint = "movie/popular"  # Esempio: film popolari
    max_pages = 2  # Numero massimo di pagine da ottenere
    path_csv = "./dataset/"

    TMDbAPI = TMDbAPI(api_key)

    complete_path = path_csv + "Popular_film.csv"

    if checking_path(path_csv) is False:
        movie_dataframe = TMDbAPI.fetch_movie_data(endpoint, max_pages)
        movie_dataframe.to_csv(complete_path, index = False)

    df = pd.read_csv("./dataset/Popular_film.csv")
    print(df.info())
    print(df['genres'])
