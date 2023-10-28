import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import scipy.sparse

def evaluation_models():
    # Carica il tuo dataset
    data = pd.read_csv('normalized_Popular_film.csv')

    # Preprocessing dei dati
    # Gestisci dati mancanti (se presenti)
    data.dropna(inplace=True)

    # Seleziona le altre feature numeriche per l'addestramento del modello
    numeric_features = ['popularity', 'vote_average', 'vote_count', 'likeable']
    X_numeric = data[numeric_features[:-1]]  # Escludi la colonna "likeable" come feature

    # Normalizzazione delle feature numeriche
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X_numeric)

    # Unisci la rappresentazione TF-IDF con le feature numeriche
    X_combined = scipy.sparse.hstack((tfidf_matrix, X_numeric))

    # Prepara il target
    y = data['likeable']

    # Suddividi il dataset in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Crea un classificatore KNN
    knn = KNeighborsClassifier(n_neighbors=5)  # Imposta il numero di vicini desiderati

    # Addestra il modello
    knn.fit(X_train, y_train)

    # Calcola la curva ROC e la curva di validazione
    fpr, tpr, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    train_fpr, train_tpr, _ = roc_curve(y_train, knn.predict_proba(X_train)[:, 1])
    train_auc = auc(train_fpr, train_tpr)

    # Mostra la curva ROC e la curva di validazione
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot(train_fpr, train_tpr, color='blue', lw=2, label='Curva di Validazione (area = %0.2f)' % train_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasso di falsi positivi')
    plt.ylabel('Tasso di veri positivi')
    plt.title('Curva ROC e Curva di Validazione')
    plt.legend(loc="lower right")

    # Effettua previsioni sui dati dei film
    film_predictions = knn.predict(X_combined)  # Utilizza il modello KNN addestrato su tutti i dati dei film

    # Aggiungi le previsioni al tuo dataframe dei dati
    data['predicted_likeable'] = film_predictions

    # Seleziona i film che possono piacere
    films_to_like = data[data['predicted_likeable'] == 1]

    # Mostra i film che possono piacere
    print("Film che possono piacere:")
    print(films_to_like['title', 'likeable'])

    plt.show()


