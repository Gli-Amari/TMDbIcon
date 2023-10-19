
:-style_check(-singleton).

% fatti sulla media dei voti
vote_average()

% fatti sui film
film('Movie1', 0.666667, 0.148868, 1.000000).
film('Movie2', 0.565658, 0.145263, 0.991558).
film('Movie3', 8.2, 800, 60).
film('Movie4', 5.5, 200, 30).

% Regola: Calcola una feature di gradimento basata su vote_average, vote_count, e popularity
gradimento(NomeFilm, Gradimento) :-
    film(NomeFilm, VoteAverage, VoteCount, Popularity),
    Gradimento is ((VoteAverage * 0.3) + (VoteCount * 0.3) + (Popularity * 0.4)) * 0.2.

% Regola: Calcola il valore di gradimento (0 o 1) basato sul gradimento
valore_gradimento(NomeFilm, Valore) :-
    gradimento(NomeFilm, Gradimento),
    (Gradimento > 0.5 -> Valore = 1 ; Valore = 0).

% Esempi di utilizzo
film_gradito(NomeFilm) :-
    valore_gradimento(NomeFilm, 1).

film_non_gradito(NomeFilm) :-
    valore_gradimento(NomeFilm, 0).
