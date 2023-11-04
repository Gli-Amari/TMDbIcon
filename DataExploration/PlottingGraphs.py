import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import stopwords


class PlottingGraphs:
    def __init__(self, df):
        self.df: pd.DataFrame = df

    def frequent_keywords(self, column_name='keywords', top_n=20):
        # Rimozione dei caratteri '[', ']', e "'" dalle keywords
        all_keywords = self.df[column_name].dropna().str.replace('[', '').str.replace(']', '').str.replace("'",'').str.split(', ')

        # Unione di tutte le keywords in una lista e conteggio della frequenza
        all_keywords = [keyword for sublist in all_keywords for keyword in sublist]
        keyword_count = Counter(all_keywords)
        top_keywords = dict(keyword_count.most_common(top_n))

        # Plot del grafico a barre delle top keywords
        plt.figure(figsize=(10, 6))
        plt.bar(top_keywords.keys(), top_keywords.values(), color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel('Keywords')
        plt.ylabel('Frequenza')
        plt.title(f'Top {top_n} Keywords più utilizzate nel DataFrame')
        plt.tight_layout()
        plt.show()

    def plotHitghtsDirector(self):
        plt.subplots(figsize=(12, 10))
        ax = self.df[self.df['director'] != ''].director.value_counts()[:10].sort_values(ascending=True).plot.barh(
            width=0.9, color=sns.color_palette('muted', 40))
        for i, v in enumerate(
                self.df[self.df['director'] != ''].director.value_counts()[:10].sort_values(ascending=True).values):
            ax.text(.5, i, v, fontsize=12, color='white', weight='bold')
        plt.title('REGISTI CON PIU APPARENCE')
        plt.show()


    def plotHightsData(self, col_name, title_graph):
        self.df[col_name] = self.df[col_name].str.strip('[]').str.replace(' ', '').str.replace("'", '')
        self.df[col_name] = self.df[col_name].str.split(',')

        plt.subplots(figsize=(12, 10))
        list1 = []
        for i in self.df[col_name]:
            list1.extend(i)
        ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,
                                                                                        color=sns.color_palette('hls',
                                                                                                                10))
        for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):
            ax.text(.8, i, v, fontsize=12, color='white', weight='bold')
        plt.title(title_graph)
        plt.show()
        return plt

    def plotHightsGenres(self, col_name, title_graph):
        self.df[col_name] = self.df[col_name].str.strip('[]').str.replace(' ', '').str.replace("'", '')
        self.df[col_name] = self.df[col_name].str.split(',')

        plt.figure(figsize=(10, 8))
        list1 = []
        for i in self.df[col_name]:
            list1.extend(i)
        top_10_values = pd.Series(list1).value_counts()[:10]

        labels = top_10_values.index
        sizes = top_10_values.values

        # Plot del grafico a torta
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Assicura che il grafico sia un cerchio piuttosto che un'ellisse
        plt.title(title_graph)
        plt.show()