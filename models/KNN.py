import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, validation_curve, learning_curve, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer  # Aggiunto SimpleImputer per valorizzare i NaN
import pandas as pd


class KNN:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.knn = KNeighborsClassifier()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluation_models(self, seed):
        imputer = SimpleImputer(strategy='mean')
        self.x_train = imputer.fit_transform(self.x_train)
        self.x_test = imputer.transform(self.x_test)

        self.knn.fit(self.x_train, self.y_train)
        param_grid = dict(n_neighbors=list(range(1, 200)))

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.knn, param_grid, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.x_train, self.y_train)
        best_model: KNeighborsClassifier = self.knn.set_params(**grid.best_params_)

        parameter_range = np.arange(1, 30, 1)

        X = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test))

        score, train_scores, valid_scores = learning_curve(estimator=best_model,
                                                           X=X, y=y,
                                                           scoring='accuracy')
        mean_train_score = np.mean(train_scores, axis=1)
        mean_valuation_score = np.mean(valid_scores, axis=1)

        # curva di apprendimento
        plt.title('curva di apprendimento')
        plt.plot(score, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training Accuracy')
        plt.plot(score, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()
        train_scores, valid_scores = validation_curve(estimator=best_model,
                                                      X=self.x_train, y=self.y_train,
                                                      param_name='n_neighbors',
                                                      param_range=parameter_range,
                                                      scoring='accuracy',
                                                      n_jobs=-1)
        mean_train_score = np.mean(train_scores, axis=1)
        mean_valuation_score = np.mean(valid_scores, axis=1)

        # curva di validazione del modello
        plt.title('curva di validazione')
        plt.plot(parameter_range, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training Accuracy')
        plt.plot(parameter_range, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation Accuracy')
        plt.xlabel('n neighbors')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_test)

        # Lista di film basata sulle predizioni del modello KNN
        film_list = []
        for i in range(len(y_pred)):
            if y_pred[i] == 1:  # Supponendo che 1 rappresenti la classe dei film che desideri includere nella lista
                film_list.append(y_pred[i])  # Aggiungi il nome del film alla lista

        # Se la lista di film è più lunga di 30, prendi solo i primi 30 film
        film_list = film_list[:30]

        # Stampa la lista di film
        print(y_pred)

        print("Report K-NN di classificazione: ")
        print(classification_report(y_pred, self.y_test))
