import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, learning_curve


class RegressioneLineare:
    def __init__(self, x_train, x_test, y_train, y_test, df_titles):
        self.linear_reg = LinearRegression()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_titles: pd.DataFrame = df_titles

    def evaluate_model(self, seed):
        imputer = SimpleImputer(strategy='mean')
        self.x_train = imputer.fit_transform(self.x_train)
        self.x_test = imputer.transform(self.x_test)

        param_grid = dict(n_jobs=list(range(1, 10)))

        # Creo uno schema di cross validazione mischiata e stratificata grazie alla seguente funzione (voglio k=5):
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        gcv = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=cv, verbose=True, n_jobs=1)

        gcv.fit(self.x_train, self.y_train)

        best_model: LinearRegression = self.linear_reg.set_params(**gcv.best_params_)

        # Preparo la partizione train-cross val:
        train_ind = np.ones(self.x_train.shape[0])
        train_ind = -train_ind
        val_ind = np.zeros(self.x_test.shape[0])
        ps = sklearn.model_selection.PredefinedSplit(test_fold=np.concatenate((train_ind, val_ind)))

        # La funzione vuole che passiamo train e val set assieme:
        X = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test))
        # Ho creato una lista di indici con -1 per tutti gli indici del train e 0 per quelli del validation set.
        # Uso "learning_curve" di sklearn, specificando lo stimatore (ovvero il modello):
        train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=ps,
                                                                scoring="max_error",
                                                                train_sizes=np.linspace(0.1, 1.0, 50),
                                                                verbose=True,
                                                                n_jobs=-1)
        # Valutiamo l'andamento medio della curva sia su train che validation set:
        plt.plot(train_sizes, train_scores, color="blue", label="Training set")
        plt.plot(train_sizes, test_scores, color="darkorange", label="Cross-validation set")

        plt.title("LEARNING CURVE\n(Regressione lineare)\n", color='red', fontsize=20)
        plt.xlabel("\nDimensioni del training set\n", color='black', fontsize=15)
        plt.ylabel("\nMax error\n", color='black', fontsize=15), plt.legend(loc="best")
        plt.grid()
        plt.show()

        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_test)

        print("R2 Score della Regressione lineare :", np.abs(r2_score(self.y_test, y_pred)))
        print("MAE :", mean_absolute_error(self.y_test,y_pred))
        print("MSE :", mean_squared_error(self.y_test,y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))

        ypred = pd.DataFrame({'predicted_likeable': y_pred})
        films_to_like = pd.concat([self.df_titles, ypred], axis=1)
        filtered_df = films_to_like[films_to_like['predicted_likeable'] == 1]

        # Mostra i film che possono piacere
        print("Film che possono piacere:")
        print(films_to_like.columns)
        print(filtered_df[['title', 'predicted_likeable']].head(20))