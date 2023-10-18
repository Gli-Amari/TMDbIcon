% Definizione di una lista di film con le relative feature
film('Movie1', 8, 4757, 0.005375).
film('Movie2', 6.8, 300, 45).
film('Movie3', 8.2, 800, 60).
film('Movie4', 5.5, 200, 30).

% Regola: Calcola una feature di gradimento basata su vote_average, vote_count, e popularity
gradimento(NomeFilm, Gradimento) :-
    film(NomeFilm, VoteAverage, VoteCount, Popularity),
    Gradimento is ((VoteAverage * 0.3) + (VoteCount * 0.3) + (Popularity * 0.4) * 0.2).

% Regola: Calcola il valore di gradimento (0 o 1) basato sul gradimento
valore_gradimento(NomeFilm, Valore) :-
    gradimento(NomeFilm, Gradimento),
    (Gradimento < 30 -> Valore = 1 ; Valore = 0).

% Esempi di utilizzo
% Film graditi con un gradimento maggiore di 7.0
film_gradito(NomeFilm) :-
    valore_gradimento(NomeFilm, 1).

% Film non graditi con un gradimento minore o uguale a 5.0
film_non_gradito(NomeFilm) :-
    valore_gradimento(NomeFilm, 0).
