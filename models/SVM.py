import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


class SVM:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.svm = SVC()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self, seed):
        self.svm.fit(self.x_train, self.y_train)

        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.svm, param_grid, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.x_train, self.y_train)

        best_model: SVC = self.svm.set_params(**grid.best_params_)

        parameter_range = np.arange(1, 25, 1)

        score, train_scores, valid_scores = learning_curve(estimator=best_model,
                                                           X=self.x_train, y=self.y_train,
                                                           scoring='accuracy')

        std_train_score = np.mean(train_scores, axis=1)

        std_valuation_score = np.mean(valid_scores, axis=1)
        plt.title('Curva di apprendimento')
        plt.plot(score, std_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training Accuracy')
        plt.plot(score, std_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        train_scores, valid_scores = validation_curve(estimator=best_model,
                                                      X=self.x_train, y=self.y_train,
                                                      param_name='C',
                                                      param_range=parameter_range,
                                                      scoring='accuracy',
                                                      n_jobs=-1)
        mean_train_score = np.mean(train_scores, axis=1)

        mean_valuation_score = np.mean(valid_scores, axis=1)

        plt.title('Curva di validazione')

        plt.plot(parameter_range, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training Accuracy')
        plt.plot(parameter_range, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation Accuracy')

        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_test)

        StdS_X = StandardScaler()
        StdS_y = StandardScaler()

        # inverse the transformation to go back to the initial scale
        plt.scatter(StdS_X.inverse_transform(self.x_test), StdS_y.inverse_transform(y_pred), color='red')
        plt.plot(StdS_X.inverse_transform(self.x_test), StdS_y.inverse_transform(best_model.predict(self.x_test).reshape(-1, 1)),
                 color='blue')
        # add the title to the plot
        plt.title('Support Vector Regression Model')
        # label x axis
        plt.xlabel('Position')
        # label y axis
        plt.ylabel('Salary Level')
        # print the plot
        plt.show()

        print("REPORT DEL MIGLIORE MODELLO SVM TROVATO")
        print(classification_report(y_pred, self.y_test))
