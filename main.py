
from processing.TMDbAPI import TMDbAPI
from processing.Proccessing import Processing
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def checking_path(path_csv):
    check_path = Path(path_csv)
    if check_path.is_dir():
        complete_path = Path(path_csv + "normalized_Popular_film.csv")
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

    # converto le feature booleane in feature intere
    movie_dataframe['adult'] = movie_dataframe['adult'].astype(int)

    # converto le feature categoriche in feature intere (un film puo essere rilasciato o meno per ese.)
    movie_dataframe['status'] = movie_dataframe['status'].replace('Released', 1)
    movie_dataframe['status'] = movie_dataframe['status'].replace('Not Released', 0)
    movie_dataframe['status'] = movie_dataframe['status'].replace('Post Production', 2)
    movie_dataframe['status'] = movie_dataframe['status'].replace('In Production', 2)
    movie_dataframe['status'] = movie_dataframe['status'].replace('Planned', 3)
    movie_dataframe['status'] = movie_dataframe['status'].astype(int)

    #salvo in movie_Dataframe['genres'] il primo nome rilevante della lista di generi associata
    processing.getFirstValueFromFeature('genres')
    processing.getFirstValueFromFeature('spoken_languages')
    processing.getFirstValueFromFeature('production_countries')
    processing.getFirstValueFromFeature('production_companies')

    # minmax scaler (dubbio se normalizzare prima o normalizzare dopo il KB!)
    movie_dataframe['vote_average'] = movie_dataframe['vote_average'].round().astype(int)
    scaler = MinMaxScaler()
    movie_dataframe['popularity'] = scaler.fit_transform(movie_dataframe[['popularity']])
    movie_dataframe['vote_average'] = scaler.fit_transform(movie_dataframe[['vote_average']])
    movie_dataframe['vote_count'] = scaler.fit_transform(movie_dataframe[['vote_count']])

    processing.create_feature_target()

    movie_dataframe = movie_dataframe.dropna() #cancella righe che contengono NaNg

    return movie_dataframe


if __name__ == "__main__":

    api_key = '293e12b22f35ee4b22ee998909252150'
    endpoint = "movie/popular"  # Esempio: film popolari
    max_pages = 2  # Numero massimo di pagine da ottenere
    path_csv = "./dataset/"

    # estrazione dati da server TMDb
    TMDbAPI = TMDbAPI(api_key)

    complete_path = path_csv + "normalized_Popular_film.csv"
    if checking_path(path_csv) is False:
        movie_dataframe = TMDbAPI.fetch_movie_data(endpoint, max_pages)

        # processing dataset Popular_film
        processingPopularFilmDataset(movie_dataframe)

        movie_dataframe.to_csv(complete_path, index=False)

    df = pd.read_csv("./dataset/normalized_Popular_film.csv")

    # Supponiamo che tu abbia già un dataframe df e la lista delle selected features
    selected_features = ['title', 'popularity', 'vote_average', 'vote_count', 'Likeable']
    selected_data = df[selected_features]

    # Genera un campione casuale di 5 righe
    sample_data = selected_data.sample(n=5, random_state=20)

    # Ordina il campione in base a 'popularity', 'vote_count' e 'vote_average'
    sample_data_sorted = sample_data.sort_values(by=['popularity', 'vote_count', 'vote_average'],
                                                 ascending=[False, False, False])

    # Stampa il campione ordinato
    print(sample_data_sorted)




