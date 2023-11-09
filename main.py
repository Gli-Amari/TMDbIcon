import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from DataExploration.PlottingGraphs import PlottingGraphs
from sklearn.model_selection import train_test_split
from models.MLPRegressorModel import MLPRegressorModel
from COP.SimulateAnneliening import COPRecommending
from KnowledgEngigne.KnowledgEngigne import KnwoledgeEngine
from models.KNNmodel import KNNmodel
from processing.Proccessing import Processing
from models.svrModel import svrModel
from models.linearRegressionModel import LinearRegressionModel


def checking_path(path_csv):
    check_path = Path(path_csv)
    if check_path.is_dir():
        complete_path = Path(path_csv + "normalized_Popular_film.csv")
        if complete_path.exists():
            return True
        else:
            return False


def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']


def processingDataset(complete_path, path_csv):
    movie_dataframe = pd.read_csv("./dataset/tmdb_5000_movies.csv")
    credits_dataframe = pd.read_csv("./dataset/tmdb_5000_credits.csv")

    processing = Processing(df_movies=movie_dataframe, df_credits=credits_dataframe, df=None)
    processing.convertJSONToStringMovies(col_name='genres', value_index='name')
    processing.convertJSONToStringMovies(col_name='keywords', value_index='name')
    processing.convertJSONToStringMovies(col_name='production_countries', value_index='iso_3166_1')
    processing.convertJSONToStringMovies(col_name='production_companies', value_index='name')
    processing.convertJSONToStringMovies(col_name='spoken_languages', value_index='iso_639_1')

    processing.convertJSONToStringCredits(col_name='cast', value_index='name')

    credits = processing.getDfCredits()
    movies = processing.getDfMovies()

    credits['crew'] = credits['crew'].apply(json.loads)
    credits['crew'] = credits['crew'].apply(director)
    credits.rename(columns={'crew': 'director'}, inplace=True)

    daataframe_merged = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
    daataframe_merged = daataframe_merged[['original_title', 'original_language',
                                           'overview', 'genres',
                                           'cast', 'keywords',
                                           'director', 'status', 'vote_average',
                                           'vote_count', 'popularity']]

    for i, j in zip(daataframe_merged['genres'], daataframe_merged.index):
        list2 = i.strip('[]').replace(' ', '').replace("'", '').split(',')
        list2.sort()
        daataframe_merged.at[j, 'genres'] = str(list2)

    daataframe_merged['genres'] = daataframe_merged['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    daataframe_merged['genres'] = daataframe_merged['genres'].str.split(',')

    if checking_path(path_csv) is False:
        daataframe_merged.to_csv(complete_path, index=False)

def processingForSupervizedLearning(complete_path, path_csv):
    movie_dataframe = pd.read_csv("./dataset/datasetMerged.csv")
    processing = Processing(df_movies=None, df_credits=None, df=movie_dataframe)

    processing.LabelEncoding(col_name='genres')
    processing.LabelEncoding(col_name='cast')
    processing.LabelEncoding(col_name='keywords')
    processing.LabelEncoding(col_name='status')
    processing.LabelEncoding(col_name='director')
    processing.LabelEncoding(col_name='original_language')

    processing.minMaxScaler(col_name='vote_average')
    processing.minMaxScaler(col_name='vote_count')
    processing.minMaxScaler(col_name='popularity')
    processing.minMaxScaler(col_name='director')
    processing.minMaxScaler(col_name='keywords')
    processing.minMaxScaler(col_name='cast')
    processing.minMaxScaler(col_name='genres')
    processing.minMaxScaler(col_name='original_language')

    print(movie_dataframe)
    if checking_path(path_csv) is False:
        movie_dataframe.to_csv(complete_path, index=False)


if __name__ == "__main__":
    path_csv = "./dataset/"
    file_path = path_csv + "datasetMerged.csv"
    if os.path.isfile(file_path) is False:
        complete_path = path_csv + "datasetMerged.csv"
        processingDataset(complete_path, path_csv)

    df = pd.read_csv("./dataset/datasetMerged.csv")

    print("Data Exploration")
    plotter = PlottingGraphs(df)
    plotter.frequent_keywords(column_name='keywords')
    plotter.plotHightsGenres(col_name='genres', title_graph='Gneneri più frequenti')
    plotter.plotHightsData(col_name='cast', title_graph='Attori più frequenti')
    plotter.plotHistVoteAv()

    print("Reasoning by inference")
    path_csv = "./dataset/"
    file_path = path_csv + "normalizedDataset.csv"
    if os.path.isfile(file_path) is False:
        complete_path = path_csv + "normalizedDataset.csv"
        processingForSupervizedLearning(complete_path, path_csv)

    df = pd.read_csv("./dataset/normalizedDataset.csv")

    knowledgeEngine = KnwoledgeEngine(df)
    knowledgeEngine.createFacts()
    learningDF = knowledgeEngine.KBInterrogation()
    learningDF.to_csv("./dataset/learningDataset.csv")

    learningDataset = pd.read_csv("./dataset/learningDataset.csv")

    print("Top 10 film by inference: ")
    learningDataset = learningDataset.sort_values(by='ratio_likeable', ascending=False)
    firts_10_values = learningDataset[['original_title', 'ratio_likeable']].head(10)
    print(firts_10_values)

    firts_10_values.to_csv("./rankingResult/ranking_by_inference.csv")

    print("Machine Learning regression:")

    column_to_delet = ['original_title', 'overview']
    df_apart = learningDataset[column_to_delet].copy()
    learningDataset.drop(column_to_delet, axis=1, inplace=True)
    learningDataset.drop('Unnamed: 0', axis=1, inplace=True)

    column_to_delet = ['ratio_likeable']
    df_apart_likeable = learningDataset[column_to_delet].copy()
    learningDataset.drop(column_to_delet, axis=1, inplace=True)
    X = learningDataset
    y = df_apart_likeable

    y = np.array(y)
    y = np.ravel(y)

    seed = 53

    x_train, x_test, y_train, y_test = train_test_split(X, np.round(y),
                                                        test_size=0.30,
                                                        train_size=0.70,
                                                        shuffle=True, random_state=seed)

    print("SVR model result")
    svrModel(x_train, x_test, y_train, y_test, df_apart).evaluate_model(seed)
    print("MLP Regressor model result")
    MLPRegressorModel(x_train, x_test, y_train, y_test, df_apart).evaluation_models(seed)
    print("LinearRegression model result")
    LinearRegressionModel(x_train, x_test, y_train, y_test, df_apart).evaluate_model(seed)

    print("COP problem result")
    file_path = './dataset/learningDataset.csv'
    csp_recommendation = COPRecommending(file_path)
    selected_films = csp_recommendation.recommend_films(k=10)
    df_bestSelection_films = pd.DataFrame(selected_films)
    df_bestSelection_films.to_csv("./rankingResult/ranking_by_COP.csv")
    print(df_bestSelection_films)

    print("machine learning vlass")
    df = pd.read_csv('./dataset/normalizedDataset.csv')
    lunghezza_dataframe = len(df)
    df['nuova_colonna'] = np.random.choice([0, 1], size=lunghezza_dataframe)
    if len(df['nuova_colonna']) != lunghezza_dataframe:
        print("Errore: la lunghezza della nuova colonna non corrisponde alla lunghezza del DataFrame.")
    else:
        df.to_csv('normalizedDataset_con_nuova_colonna.csv', index=False)

    column_to_delet = ['original_title', 'overview']
    df_apart = df[column_to_delet].copy()
    df.drop(column_to_delet, axis=1, inplace=True)
    # df.drop('Unnamed: 0', axis=1, inplace=True)
    column_to_delet = ['nuova_colonna']
    df_apart_likeable = df[column_to_delet].copy()
    df.drop(column_to_delet, axis=1, inplace=True)
    X = df
    y = df_apart_likeable
    seed = 53
    y = np.array(y)
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        train_size=0.70,
                                                        shuffle=True, random_state=seed)

    print("KNN Classifier model result")
    KNNmodel(x_train, x_test, y_train, y_test, df_apart).evaluation_models(seed)






