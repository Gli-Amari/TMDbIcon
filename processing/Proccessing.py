
from pyswip import Prolog

class Processing:

    def __init__(self, df):
        self.df = df

    def extraction(self, col_name, key):
        self.df[col_name] = self.df[col_name].apply(lambda x: [item[key] for item in x] if x else None)
        return self.df

    def create_feature_target(self):

        prolog = Prolog()
        prolog.consult("\prolog\supervized_KB.pl.pl")
        result = []

        for _, val in self.df.iterrows():
            query = (f"gradimento(film({str(val['title']).replace(' ', '')},"
                     f"{val['vote_average']},{str(val['vote_count']).replace(' ', '')},"
                     f"{val['popularity']},"
                     f"), Gradimento)")
            result.append(bool(list(prolog.query(query))))

        self.df['Likeable'] = result

    def getFirstValueFromFeature(self, col_name):
        self.df[col_name] = self.df[col_name].astype(str)
        self.df[col_name] = self.df[col_name].str.replace(r'\[|\]|,\s*$', '', regex=True)
        self.df[col_name] = self.df[col_name].str.split(',').str.get(0)
        self.df[col_name] = self.df[col_name].str.replace("'", "")
        return self.df