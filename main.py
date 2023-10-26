import os

from processing.TMDbAPI import TMDbAPI
from processing.Proccessing import Processing
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from models.KNN import KNN


def checking_path(path_csv):
    check_path = Path(path_csv)
    if check_path.is_dir():
        complete_path = Path(path_csv + "normalized_Popular_film.csv")
        if complete_path.exists():
            return True
        else:
            return False


# richiamo in una routine esclusivamente le op. per fare processing su dataset Popular_film!
def processingPopularFilmDataset(complete_path, path_csv):
    movie_dataframe = TMDbAPI.fetch_movie_data(endpoint, max_pages)
    processing = Processing(movie_dataframe)

    # sistemiamo le feature che hanno al loro interno piu valori (un film puo aver piu generi per ese.)
    processing.extraction(col_name='genres', key='id')
    processing.extraction(col_name='production_companies', key='id')
    processing.extraction(col_name='production_countries', key='iso_3166_1')
    processing.extraction(col_name='spoken_languages', key='iso_639_1')

    # converto le feature booleane in feature intere
    processing.convertFeatureToInt(col_name='adult')

    # converto le feature categoriche in feature intere (un film puo essere rilasciato o meno per ese.)
    processing.replaceFeatureValue(col_name='status', valueToReplace='Released', value=1)
    processing.replaceFeatureValue(col_name='status', valueToReplace='Not Released', value=0)
    processing.replaceFeatureValue(col_name='status', valueToReplace='Post Production', value=2)
    processing.replaceFeatureValue(col_name='status', valueToReplace='In Production', value=3)
    processing.replaceFeatureValue(col_name='status', valueToReplace='Planned', value=3)

    processing.convertFeatureToInt(col_name='status')

    # salvo in movie_Dataframe['genres'] il primo nome rilevante della lista di generi associata
    processing.getFirstValueFromFeature('genres')
    processing.getFirstValueFromFeature('spoken_languages')
    processing.getFirstValueFromFeature('production_countries')
    processing.getFirstValueFromFeature('production_companies')

    processing.replaceIso3166(col_name='production_countries')
    processing.replaceIso639(col_name='spoken_languages')
    processing.replaceIso639(col_name='original_language')

    # minmax scaler (dubbio se normalizzare prima o normalizzare dopo il KB!)
    movie_dataframe['vote_average'] = movie_dataframe['vote_average'].round().astype(int)

    processing.minMaxScaler(col_name='vote_average')
    processing.minMaxScaler(col_name='vote_count')
    processing.minMaxScaler(col_name='popularity')

    processing.dropNaN()

    normalized_df = processing.KBInterrogation()

    if checking_path(path_csv) is False:
        normalized_df.to_csv(complete_path, index=False)


if __name__ == "__main__":
    api_key = '293e12b22f35ee4b22ee998909252150'
    endpoint = "movie/popular"  # Esempio: film popolari
    max_pages = 150  # Numero massimo di pagine da ottenere
    path_csv = "./dataset/"

    # estrazione dati da server TMDb
    file_path = path_csv + "normalized_Popular_film.csv"
    if os.path.isfile(file_path) is False:
        TMDbAPI = TMDbAPI(api_key)
        complete_path = path_csv + "normalized_Popular_film.csv"
        processingPopularFilmDataset(complete_path, path_csv)

    seed = 53
    df = pd.read_csv("./dataset/normalized_Popular_film.csv")
    X = df
    Y = df['likeable']
    X = X.drop('likeable', axis=1)
    X_train, X_test, Y_train, Y_set = train_test_split(X, Y, stratify=Y, test_size=0.30, train_size=0.70, shuffle=True,
                                                       random_state=seed)
    print("Risultati ottenuti dai modelli")
    KNN(X_train, X_test, Y_train, Y_set).evaluation_models(seed)