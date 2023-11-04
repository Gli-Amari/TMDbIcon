import csv
import pandas as pd
from pyswip import Prolog


class KnwoledgeEngine:
    def __init__(self, df):
        self.df: pd.DataFrame = df

    def getFactsVoteC(self, row):
        return 'hasVoteC("{}",hasVoteC,{}).'.format(row[0], row[8])

    def getFactsVoteAv(self, row):
        return 'hasVoteAv("{}",hasVoteAv,{}).'.format(row[0], row[9])

    def getFactsPop(self, row):
        return 'hasPop("{}",hasPop,{}).'.format(row[0], row[10])

    
    def createFacts(self):
        predicate = ":- discontiguous hasVoteC/3. " + "\n" + ":- discontiguous hasVoteAv/3." + "\n" + ":- discontiguous hasPop/3." + "\n\n"

        with open('./dataset/normalizedDataset.csv', 'r', encoding='utf-8') as csv_file, open('factsKB.pl', 'w',
                                                                                              encoding='utf-8') as prolog_file:
            prolog_file.write(predicate)
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Salta la riga d'intestazione
            for row in csv_reader:
                prolog_factVoteAv = self.getFactsVoteAv(row)
                prolog_factVoteAC = self.getFactsVoteC(row)
                prolog_factPop = self.getFactsPop(row)
                prolog_file.write(prolog_factVoteAv + '\n'
                                  + prolog_factVoteAC + '\n'
                                  + prolog_factPop + '\n \n')

    def KBInterrogation(self):
        prolog = Prolog()
        prolog.consult('./supervized_KB.pl')
        result = []
        for _, row in self.df.iterrows():
            assertz = ('ratio(' + str(row['vote_average'])
                       + ',' + str(row['vote_count']) + ',' + str(
                        row['popularity']) + ',' + 'Ratio).')
            result.append(list(prolog.query(assertz))[0]['Ratio'])

        self.df['ratio_likeable'] = result
        return self.df





