import json
import os
import pandas as pd
from pathlib import Path
from DataExploration.PlottingGraphs import PlottingGraphs
from sklearn.model_selection import train_test_split
from models.DecisionTreeClassifier import DecisionTreeClassifier, MyDecisionTreeClassifier
from models.KNN import KNN
from models.RandomForest import RandomForest
from models.LinearRegression import RegressioneLineare
from processing.Proccessing import Processing


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
    daataframe_merged = daataframe_merged[['id', 'original_title', 'original_language',
                                           'overview', 'genres',
                                           'cast', 'keywords',
                                           'director', 'status', 'vote_average',
                                           'vote_count', 'popularity']]

    # Sort the genres in each row
    for i, j in zip(daataframe_merged['genres'], daataframe_merged.index):
        list2 = i.strip('[]').replace(' ', '').replace("'", '').split(',')
        list2.sort()
        daataframe_merged.at[j, 'genres'] = str(list2)

    # Split the sorted strings into lists
    daataframe_merged['genres'] = daataframe_merged['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    daataframe_merged['genres'] = daataframe_merged['genres'].str.split(',')

    if checking_path(path_csv) is False:
        daataframe_merged.to_csv(complete_path, index=False)


def processingForSupervizedLearning(complete_path, path_csv):
    movie_dataframe = pd.read_csv("./dataset/datasetMerged.csv")
    processing = Processing(df_movies=None, df_credits=None, df=movie_dataframe)

    #generate a list 'genreList' with all possible unique genres mentioned in the dataset for OneHotEncoding
    movie_dataframe['genres'] = movie_dataframe['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    movie_dataframe['genres'] = movie_dataframe['genres'].str.split(',')
    genreList = []
    for index, row in movie_dataframe.iterrows():
        genres = row["genres"]
        for genre in genres:
            if genre not in genreList:
                genreList.append(genre)
    movie_dataframe['genres_bin'] = movie_dataframe['genres'].apply(lambda x: processing.oneHotEncoding(x))

    #generate a list 'castlist' with all possible unique genres mentioned in the dataset for OneHotEncoding (salva lo stesso OneHot!! prov a ad usare la funzione che sai usare)
    for i, j in zip(movie_dataframe['cast'], movie_dataframe.index):
        list2 = []
        list2 = i[:4]
        movie_dataframe.loc[j, 'cast'] = str(list2)
    movie_dataframe['cast'] = movie_dataframe['cast'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    movie_dataframe['cast'] = movie_dataframe['cast'].str.split(',')
    for i, j in zip(movie_dataframe['cast'], movie_dataframe.index):
        list2 = []
        list2 = i
        list2.sort()
        movie_dataframe.loc[j, 'cast'] = str(list2)
    movie_dataframe['cast'] = movie_dataframe['cast'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    castList = []
    for index, row in movie_dataframe.iterrows():
        cast = row["cast"]
        for i in cast:
            if i not in castList:
                castList.append(i)
    movie_dataframe['cast_bin'] = movie_dataframe['cast'].apply(lambda x: processing.oneHotEncoding(x))

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
    plotter.plotHightsData(col_name='genres', title_graph='Untilted 1')
    plotter.plotHightsData(col_name='cast', title_graph='Untilted 2')
    plotter.plotHitghtsDirector()
    # aggiungere wordcloud!

    print("Reasoning by inference")
    path_csv = "./dataset/"
    file_path = path_csv + "normalizedDataset.csv"
    if os.path.isfile(file_path) is False:
        complete_path = path_csv + "normalizedDataset.csv"
        processingForSupervizedLearning(complete_path, path_csv)

    df = pd.read_csv("./dataset/normalizedDataset.csv")
    # task

    # modelli di apprendimento supervisionato
    '''
    seed = 53
    df_titles = df['title']
    df = df.drop('title', axis=1)
    X = df
    Y = df['likeable']
    X = X.drop('likeable', axis=1)
    '''

    # x_train, x_test, y_train_reg, y_test_reg = train_test_split(X, Y,
    # stratify=round(Y),
    # test_size=0.30,
    # train_size=0.70,
    # shuffle=True, random_state=seed)
    # y_train = round(y_train_reg)
    # y_test = round(y_test_reg)

    # RandomForest(x_train, x_test, y_train, y_test, df_titles).evaluation_models(seed)
    # MyDecisionTreeClassifier(x_train, x_test, y_train, y_test).evaluation_model(seed, 'decisionTree.dot')
    # KNN(x_train, x_test, y_train, y_test).evaluation_model(seed)
    # RegressioneLineare(x_train, x_test, y_train_reg, y_test_reg, df_titles).evaluate_model(seed)
