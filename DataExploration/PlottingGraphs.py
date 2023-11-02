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

    def plotWorldCloud(self):
        # Carica le stopwords da nltk
        stop_words = set(stopwords.words('english'))
        stop_words.update(',', ';', '!', '?', '.', '(', ')', '$', '#', '+', ':', '...', ' ', '')

        words = self.df['keywords'].dropna().str.lower().str.split()

        word_list = [word for sublist in words for word in sublist]

        filtered_words = [word for word in word_list if word not in stop_words]

        word_freq = Counter(filtered_words)

        common_words = word_freq.most_common(20)

        words, frequencies = zip(*common_words)

        plt.figure(figsize=(10, 6))
        plt.bar(words, frequencies)
        plt.xticks(rotation=45)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('PAROLE PIU USATE')
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