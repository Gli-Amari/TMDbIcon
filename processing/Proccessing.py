from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class Processing:

    transformes = [
        ['generes vectorizer', OneHotEncoder(), [1]],
        ['companies vectorizer', OneHotEncoder(), [11]],
        ['language vectorizer', OneHotEncoder(), []]
    ]
    ct = ColumnTransformer(transformes, remainder= 'passthrough')


