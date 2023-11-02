
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pyswip import Prolog
import matplotlib.pyplot as plt


class Processing:

    def __init__(self, df_movies=None, df_credits=None, df=None):
        if df_movies is None and df_credits is None:
            self.df: pd.DataFrame = df
        else:
            self.df_movies: pd.DataFrame = df_movies
            self.df_credits: pd.DataFrame = df_credits

    def oneHotEncoding(self, list_encoding):
        binaryList = []
        for genre in list_encoding:
            if genre in list_encoding:
                binaryList.append(1)
            else:
                binaryList.append(0)
        return binaryList

    def convertJSONToStringMovies(self, col_name, value_index):
        self.df_movies[col_name] = self.df_movies[col_name].apply(json.loads)
        for index, row in self.df_movies.iterrows():
            list1 = []
            for j in range(len(row[col_name])):
                list1.append(row[col_name][j][value_index])
            self.df_movies.loc[index, col_name] = str(list1)
        return self.df_movies

    def convertJSONToStringCredits(self, col_name, value_index):
        self.df_credits[col_name] = self.df_credits[col_name].apply(json.loads)
        for index, row in self.df_credits.iterrows():
            list1 = []
            for j in range(len(row[col_name])):
                list1.append(row[col_name][j][value_index])
            self.df_credits.loc[index, col_name] = str(list1)
        return self.df_credits

    def removeComma(self, col_name):
        self.df['genres'] = self.df['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
        self.df['genres'] = self.df['genres'].str.split(',')
        return self.df

    def minMaxScaler(self, col_name):
        scaler = MinMaxScaler()
        self.df[col_name] = scaler.fit_transform(self.df[[col_name]])
        return self.df

    def getDfMovies(self):
        return self.df_movies

    def getDfCredits(self):
        return self.df_credits

    def getDf(self):
        return self.df_credits

    def dropNaN(self):
        self.df.dropna(inplace=True)
        return self.df

    def KBInterrogation(self):
        prolog = Prolog()
        prolog.consult('./supervized_KB.pl')
        result = []
        for _, row in self.df.iterrows():
            assertz = ('gradimento_bilanciato(' + str(row['vote_average'])
                       + ',' + str(row['vote_count']) + ',' + str(
                        row['popularity']) + ',' + 'Ratio).')
            result.append(list(prolog.query(assertz))[0]['Ratio'])

        self.df['likeable'] = result
        return self.df

    def integer_to_string(self, col_name):
        interi_a_titoli = {indice: titolo for titolo, indice in self.df[col_name].items()}

        # Ora, mappa i valori interi nella colonna 'title' alle stringhe originali
        self.df.loc[:, col_name] = self.df[col_name].map(interi_a_titoli)
        return self.df

    def string_to_integer(self, col_name):
        titoli_univoci = self.df[col_name].unique()
        titoli_a_interi = {titolo: indice for indice, titolo in enumerate(titoli_univoci)}
        self.df[col_name] = self.df[col_name].map(titoli_a_interi)
        return self.df
