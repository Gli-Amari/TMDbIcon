
:-style_check(-singleton).

% abbiamo voluto creare i fatti su un sample di 5 esempi per dimostrare le regole

% fatti sulla media dei voti
vote_average('Sound of Freedom', 1.000000).
vote_average('Killers of the Flower Moon', 1.000000).
vote_average('Saw X', 0.666667).
vote_average('The Nun II', 0.666667).
vote_average('Fast X', 0.666667).

% fatti sul conteggio dei voti
vote_count('Sound of Freedom', 0.133923).
vote_count('Killers of the Flower Moon', 0.002612).
vote_count('Saw X', 0.022780).
vote_count('The Nun II', 0.148868).
vote_count('Fast X', 0.583720).

% fatti sulla popolarità dei film
popularity('Sound of Freedom', 0.325080).
popularity('Killers of the Flower Moon', 0.111543).
popularity('Saw X', 0.140075).
popularity('The Nun II', 0.542824).
popularity('Fast X', 0.162078).

% Regola: Calcola una feature di gradimento basata su vote_average, vote_count, e popularity
gradimento(film(NomeFilm, VoteAverage, VoteCount, Popularity), Gradimento) :-
    vote_average(NomeFilm, VoteAverage),
    vote_count(NomeFilm, VoteCount),
    popularity(NomeFilm, Popularity),
    Gradimento is ((VoteAverage * 0.3) + (VoteCount * 0.3) + (Popularity * 0.4)) * 100.

% Regola: Calcola il valore di gradimento (0 o 1) basato sul gradimento
valore_gradimento(film(NomeFilm, VoteAverage, VoteCount, Popularity), Valore) :-
    gradimento(NomeFilm, Gradimento),
    (Gradimento >= 50 -> Valore = 1 ; Valore = 0).

% Esempi di utilizzo
film_gradito(NomeFilm) :-
    valore_gradimento(NomeFilm, 1).

film_non_gradito(NomeFilm) :-
    valore_gradimento(NomeFilm, 0).
