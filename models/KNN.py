import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, validation_curve, learning_curve, RepeatedStratifiedKFold
from sklearn.metrics import classification_report

class KNN:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.knn = KNeighborsClassifier()
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    def evalutation_models(self, seed):
        self.knn.fit(self.X_train, self.Y_train)
        param_grid = dict(n_neighboors = list(range(1,100)))

        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = seed)
        grid = GridSearchCV(self.knn, param_grid, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.X_train, self.Y_train)
        best_model:KNeighborsClassifier = self.knn.set_params(**grid.best_params_)

        parameter_range = np.arange(1, 30, 1)
        X = np.concatenate((self.X_train, self.X_test), axis = 0)
        y = np.concatenate((self.Y_train, self.Y_test))

        score, train_scores, valid_scores = learning_curve(estimator = best_model, X = X, y = y, scoring = 'accuracy')
        mean_train_score = np.mean(train_scores, axis = 1)
        mean_valutation_score = np.mean(valid_scores, axis=1)
        plt.title('curava di apprendimento')
        plt.plot(score, mean_train_score, marker='o', markesize= 5, color='black', label= 'Training Accuracy')
        plt.plot(score, mean_valutation_score, marker='o', markesize=5, color='green', label='Valid Accuracy')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        train_scores, valid_scores = validation_curve(estimator=best_model, X=self.X_train, y=self.Y_train, param_name='n_neighbors', param_range = parameter_range, scoring='accuracy', n_jobs=-1)
        mean_train_score = np.mean(train_scores, axis=1)
        mean_valutation_score = np.mean(valid_scores, axis=1)

        plt.title('curva di validazione')
        plt.plot(parameter_range, mean_train_score, marker='o', markesize = 5, color= 'black', label='Training Accuracy')
        plt.plot(score, mean_valutation_score, marker='o', markesize=5, color='green', label='Valid Accuracy')
        plt.xlabel('n neighbors')
        plt.ylable('Accuracy')
        plt.grid()
        plt.show()

        best_model.fit(self.X_train, self.Y_train)
        pred = best_model.predict(self.X_test)
        print(self.knn_report(pred, self.Y_test))

