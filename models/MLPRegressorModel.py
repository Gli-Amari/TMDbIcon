from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MLPRegressorModel:
    def __init__(self, x_train, x_test, y_train, y_test, df_apart):
        self.mlp_reg = MLPRegressor()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_apart: pd.DataFrame = df_apart

    def evaluation_models(self, seed):
        self.mlp_reg.fit(self.x_train, self.y_train)
        # (10,), (20,)
        param_grid = {
            'hidden_layer_sizes': [(10,), (20,), (30,), (10, 10), (20, 10)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'max_iter': [1000],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }

        crossValidationModel = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.mlp_reg, param_grid, cv=crossValidationModel, scoring="neg_mean_squared_error",
                            error_score=0)
        grid.fit(self.x_train, self.y_train)

        bestModel: MLPRegressor = self.mlp_reg.set_params(**grid.best_params_)
        bestModel.fit(self.x_train, self.y_train)

        plt.plot(bestModel.loss_curve_)
        plt.title("Loss Curve", fontsize=14)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

        score, train_scores, valid_scores = learning_curve(estimator=bestModel,
                                                           X=self.x_train, y=self.y_train,
                                                           scoring='neg_mean_squared_error')

        mean_train_score = -np.mean(train_scores, axis=1)
        mean_valuation_score = -np.mean(valid_scores, axis=1)

        plt.title('learning curve')
        plt.plot(score, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training MSE')
        plt.plot(score, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation MSE')
        plt.ylabel('Score')
        plt.grid()
        plt.show()

        bestModel.fit(self.x_train, self.y_train)
        y_pred = bestModel.predict(self.x_test)

        df_temp = pd.DataFrame({'Predicted': y_pred})
        df_combined = pd.concat([self.df_apart, df_temp], axis=1)
        df_combined.to_csv("./rankingResult/ranking_by_MLP.csv")
        print(df_combined.head(10).sort_values(by='Predicted', ascending=True))


        df_temp = df_temp.head(30)
        df_temp.plot(kind='bar', figsize=(10, 6))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()

        print("MAE :", mean_absolute_error(self.y_test, y_pred))
        print("MSE :", mean_squared_error(self.y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))

