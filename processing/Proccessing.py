from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class Processing:

    def __init__(self, df):
        self.df = df

    def extraction(self, col_name, key):
        self.df[col_name] = self.df[col_name].apply(lambda x: [item[key] for item in x] if x else None)
        return self.df

    def getFirstValueFromFeature(self, col_name):
        self.df[col_name] = self.df[col_name].astype(str)
        self.df[col_name] = self.df[col_name].str.replace(r'\[|\]|,\s*$', '', regex=True)
        self.df[col_name] = self.df[col_name].str.split(',').str.get(0)
        self.df[col_name] = self.df[col_name].str.replace("'", "")
        return self.df