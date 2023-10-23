import csv
import sys

import pandas as pd
from pyswip import Prolog
import matplotlib.pyplot as plt

class Processing:

    def __init__(self, df):
        self.df: pd.DataFrame = df

    def extraction(self, col_name, key):
        self.df[col_name] = self.df[col_name].apply(lambda x: [item[key] for item in x] if x else None)
        return self.df

    def KBInterrogation(self):
        prolog = Prolog()
        prolog.consult('./supervized_KB.pl')
        result = []
        for _, row in self.df.iterrows():
            assertz = ('gradimento('+ str(row['vote_average'])
                       +','+ str(row['vote_count'])+','+ str(row['popularity'])+','+'Rating'+','+'Res).')
            result.append(list(prolog.query(assertz))[0]['Res'])

        self.df['Likeable'] = result
        print(self.df)

    def getFirstValueFromFeature(self, col_name):
        self.df[col_name] = self.df[col_name].astype(str)
        self.df[col_name] = self.df[col_name].str.replace(r'\[|\]|,\s*$', '', regex=True)
        self.df[col_name] = self.df[col_name].str.split(',').str.get(0)
        self.df[col_name] = self.df[col_name].str.replace("'", "")
        return self.df

    def histDataset(self):
        plt.hist(self.df, bins = 6, edgecolor = 'black', alpha = 0.7)
        plt.xlabel('Valori')
        plt.ylabel('Frequenza')
        plt.title('Esempio di Istogramma')
        return plt.show()