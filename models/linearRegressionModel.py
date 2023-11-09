import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, GridSearchCV


class LinearRegressionModel:
    def __init__(self, x_train, x_test, y_train, y_test, df_apart):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_apart = df_apart
        self.df_apart: pd.DataFrame = df_apart
        self.linearReg = LinearRegression()

    def evaluate_model(self, seed):
        paramGrid = dict(n_jobs=list(range(1, 10)))

        crossValidation = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        gcv = GridSearchCV(LinearRegression(), param_grid=paramGrid, cv=crossValidation, verbose=True, n_jobs=1)

        gcv.fit(self.x_train, self.y_train)

        bestModel: LinearRegression = self.linearReg.set_params(**gcv.best_params_)

        train_ind = -np.ones(self.x_train.shape[0])
        val_ind = np.zeros(self.x_test.shape[0])
        test_fold = np.concatenate((train_ind, val_ind))
        ps = sklearn.model_selection.PredefinedSplit(test_fold=test_fold)

        X = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test))
        train_sizes, train_scores, test_scores = learning_curve(bestModel, X, y, cv=ps,
                                                                scoring="max_error",
                                                                train_sizes=np.linspace(0.1, 1.0, 50),
                                                                verbose=True,
                                                                n_jobs=-1)
        plt.plot(train_sizes, train_scores, color="blue", label="Training set")
        plt.plot(train_sizes, test_scores, color="darkorange", label="Cross-validation set")

        plt.title("learning curve\n\n", color='red', fontsize=20)
        plt.xlabel("\ndimension of training set\n", color='black', fontsize=15)
        plt.ylabel("\nMax error\n", color='black', fontsize=15), plt.legend(loc="best")
        plt.grid()
        plt.show()

        bestModel.fit(self.x_train, self.y_train)
        y_pred = bestModel.predict(self.x_test)

        df_temp = pd.DataFrame({'Predicted': y_pred})
        df_combined = pd.concat([self.df_apart, df_temp], axis=1)
        df_combined.to_csv("./rankingResult/ranking_by_LinearRegression.csv")
        print(df_combined.head(10).sort_values(by='Predicted', ascending=True))

        print("R2 Score :", np.abs(r2_score(self.y_test, y_pred)))
        print("MAE: ", mean_absolute_error(self.y_test, y_pred))
        print("MSE: ", mean_squared_error(self.y_test, y_pred))
        print("RMSE: ", np.sqrt(mean_squared_error(self.y_test, y_pred)))