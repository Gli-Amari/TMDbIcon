import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, validation_curve, learning_curve
from sklearn.svm import SVR
import numpy as np


class svrModel:
    def __init__(self, x_train, x_test, y_train, y_test, df_apart):
        self.svr = SVR()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_apart: pd.DataFrame = df_apart

    def evaluate_model(self, seed):
        self.y_train = np.array(self.y_train)
        self.y_train = self.y_train.ravel()

        self.svr.fit(self.x_train, self.y_train)

        grid_par = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf','linear', 'poly']
        }

        cross_ValidationModels = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        gridSearchModels = GridSearchCV(self.svr, grid_par, cv=cross_ValidationModels, scoring="neg_mean_squared_error",
                                        error_score=0)
        gridSearchModels.fit(self.x_train, self.y_train)

        bestModel: SVR = self.svr.set_params(**gridSearchModels.best_params_)

        parameter_range = np.arange(1, 25, 1)

        score, trainScore, validScore = learning_curve(estimator=bestModel,
                                                       X=self.x_train,
                                                       y=self.y_train,
                                                       scoring='neg_mean_squared_error')

        std_train_score = -np.mean(trainScore, axis=1)
        std_valuation_score = -np.mean(validScore, axis=1)

        plt.title('Learning Curve')
        plt.plot(score, std_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training MSE')
        plt.plot(score, std_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation MSE')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.xlabel('Training Set Size')
        plt.grid()
        plt.legend()
        plt.show()

        trainScores, validScores = validation_curve(estimator=bestModel,
                                                    X=self.x_train, y=self.y_train,
                                                    param_name='C',
                                                    param_range=parameter_range,
                                                    scoring='neg_mean_squared_error',
                                                    n_jobs=-1)

        mean_train_score = -np.mean(trainScores, axis=1)
        mean_valuation_score = -np.mean(validScores, axis=1)

        plt.title('Validation Curve')

        plt.plot(parameter_range, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training MSE')
        plt.plot(parameter_range, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation MSE')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.xlabel('Valori del Parametro C')
        plt.grid()
        plt.show()

        bestModel.fit(self.x_train, self.y_train)
        y_pred = bestModel.predict(self.x_test)

        df_temp = pd.DataFrame({'Predicted': y_pred})
        df_combined = pd.concat([self.df_apart, df_temp], axis=1)
        df_combined.to_csv("./rankingResult/ranking_by_SVR.csv")
        print(df_combined.head(10).sort_values(by='Predicted', ascending=True))

        print("MAE :", mean_absolute_error(self.y_test, y_pred))
        print("MSE :", mean_squared_error(self.y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))
