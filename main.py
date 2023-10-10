
from processing.TMDbAPI import TMDbAPI
import pandas as pd

if __name__ == "__main__":

    api_key = '293e12b22f35ee4b22ee998909252150'
    endpoint = "movie/popular"  # Esempio: film popolari
    max_pages = 150  # Numero massimo di pagine da ottenere

    TMDbAPI = TMDbAPI(api_key)

    movie_dataframe = TMDbAPI.fetch_movie_data(endpoint, max_pages)
    movie_dataframe.to_csv("./dataset/Popular_film.csv", index = False)

    df = pd.read_csv("./dataset/Popular_film.csv")
    print(df.info())

