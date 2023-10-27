import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, learning_curve


class RandomForest:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.rfc = RandomForestClassifier()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train= y_train
        self.y_test= y_test

    def evaluation_models(self, seed):
        imputer = SimpleImputer(strategy='mean')
        self.x_train = imputer.fit_transform(self.x_train)
        self.x_test = imputer.transform(self.x_test)

        param_dist = {
            'n_estimators': np.arange(3, 25),
            'max_depth': np.arange(3, 15)
        }

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.rfc, param_dist, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.x_train, self.y_train)

        best_model: RandomForestClassifier = self.rfc.set_params(**grid.best_params_)
        best_model.fit(self.x_train, self.y_train)

        score, train_scores, valid_scores = learning_curve(estimator=best_model,
                                                           X=self.x_train, y=self.y_train,
                                                           scoring='accuracy')

        mean_train_score = np.mean(train_scores, axis=1)

        mean_valuation_score = np.mean(valid_scores, axis=1)
        plt.title('Curva di apprendimento')
        plt.plot(score, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training Accuracy')
        plt.plot(score, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_test)

        print("REPORT DELLA MIGLIOR RANDOM FOREST")
        print(classification_report(y_pred, self.y_test))